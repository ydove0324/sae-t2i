   @barrier_on_entry
    @log_on_entry
    def training_loop(self):
        initial_step = self.resume.step if self.resume else 0
        data_iterator = iter(self.train_dataloader)
        pbar = tqdm(
            iterable=range(initial_step, self.config.training.total_steps),
            disable=get_local_rank() != 0,
            initial=initial_step,
            total=self.config.training.total_steps,
        )
        accumulation = self.config.training.get("gradient_accumulation", 1)
        initial_opt_step = initial_step // accumulation

        for step in pbar:
            # Train vae.

            pbar.set_description("vae")
            is_opt_step = (step + 1) % accumulation == 0
            with autocast():
                vae_batch = next(data_iterator)
                self.mfu_start.record()
                with enable_flops_accumulate():
                    vae_losses, vae_results = self.train_vae_step(vae_batch)

            self.mfu_end.record()

            # Gradient accumulation for VAE
            (vae_losses["loss_vae"] / accumulation).backward()
            if is_opt_step:
                self.vae_optimizer.step()
                self.vae_optimizer.zero_grad()

                if self.ema_vae:
                    self.ema_vae.update()

            torch.cuda.synchronize()
            stats_mfu, stats_compute, stats_time = self.get_mfu_statistics(step)
            self.compute_accumulator.add(**stats_compute)

            # Train dis.
            # pbar.set_description("dis")
            # with autocast():
            #     dis_batch = next(data_iterator)
            #     dis_losses = self.train_dis_step(dis_batch)

            # # Gradient accumulation for Discriminator
            # (dis_losses["loss_dis"] / accumulation).backward()
            # if is_opt_step:
            #     self.dis_optimizer.step()
            #     self.dis_optimizer.zero_grad()

            # dis_losses = {}

            # Accumulate loss.
            # self.loss_accumulator.add(**dis_losses, **vae_losses)
            self.loss_accumulator.add(**vae_losses)

            # Log loss.
            if is_opt_step:
                opt_step = (step + 1) // accumulation
                if opt_step % self.config.writer.interval.loss == 0:
                    self.log_loss(opt_step)

            # Log image.
            if is_opt_step:
                opt_step = (step + 1) // accumulation
                if opt_step % self.config.writer.interval.image == 0:
                    self.log_image(opt_step, vae_batch, vae_results)

            # Save checkpoint.
            if is_opt_step:
                opt_step = (step + 1) // accumulation
                if (
                    opt_step % self.config.persistence.interval == 0
                    and opt_step != initial_opt_step
                ):
                    (
                        self.shift_factor,
                        self.scale_factor,
                        self.ema_shift_factor,
                        self.ema_scale_factor,
                    ) = self.compute_z_stats()

                    with flag_override(self.config, "readonly", False):
                        OmegaConf.update(
                            self.config, "ae.shift_factor", self.shift_factor.item(), merge=True
                        )
                        OmegaConf.update(
                            self.config, "ae.scale_factor", self.scale_factor.item(), merge=True
                        )
                        OmegaConf.update(
                            self.config,
                            "ae.ema_shift_factor",
                            self.ema_shift_factor.item(),
                            merge=True,
                        )
                        OmegaConf.update(
                            self.config,
                            "ae.ema_scale_factor",
                            self.ema_scale_factor.item(),
                            merge=True,
                        )
                    self.save_checkpoint(opt_step)

            if is_opt_step:
                opt_step = (step + 1) // accumulation
                if (
                    opt_step % self.config.evaluation.interval == 0
                    and opt_step != initial_opt_step
                ):
                    self.evaluation_loop(opt_step)

    def train_vae_step(self, batch):
        images = batch["image"].to(get_device(), memory_format=self.memory_format)
        self.add_batch_stat(images, "vae")

        batch_size = images.size(0)

        # Pass through autoencoder.
        images_pred, latents, posterior, diffusion_mse_loss = self.vae(images)

        # Pass through discriminator.
        # self.dis.requires_grad_(False)
        # logits = self.dis(images_pred)

        # Compute losses
        loss_vae_l1 = self.config.loss.l1_weight * F.l1_loss(images_pred, images)
        loss_vae_lpips = self.config.loss.lpips_weight * self.lpips(images_pred, images).mean()
        loss_vae_adversarial = torch.tensor(0.0, device=images.device)
        # loss_vae_adversarial = self.config.loss.gan_weight * gen_loss(
        #     loss_type=self.config.loss.dis_type, logits=logits
        # )
        loss_vae_kl = self.config.loss.kl_weight * posterior.kl().sum()

        loss_vae_diffusion_loss = torch.tensor(0.0, device=images.device)
        if self.config.loss.get("diffusion_mse_weight", 0) > 0:
            loss_vae_diffusion_loss = self.config.loss.diffusion_mse_weight * diffusion_mse_loss

        # Compute ref_kl loss
        loss_ref_kl = torch.tensor(0.0, device=images.device)
        if self.ref_encoder is not None:
            with torch.no_grad():
                ref_latents = self.ref_encoder(images)[0]

                cls_len = 1  # cls_token 通常只有1个
                register_len = 4  # 注册token的数量
                ref_latents = ref_latents[:, cls_len+register_len:, :].transpose(1, 2).view(batch_size, -1, 16, 16)

            loss_ref_kl = self.config.loss.ref_kl_weight * F.mse_loss(latents, ref_latents)

        # Sum all losses
        loss_vae = loss_vae_l1 + loss_vae_lpips + loss_vae_kl

        return (
            dict(
                loss_vae=loss_vae,
                loss_vae_l1=loss_vae_l1.detach(),
                loss_vae_lpips=loss_vae_lpips.detach(),
                loss_vae_adversarial=loss_vae_adversarial.detach(),
                loss_vae_kl=loss_vae_kl.detach(),
                loss_ref_kl=loss_ref_kl.detach(),
                latents_mean=latents.detach().mean(),
                latents_std=latents.detach().std(),
                loss_vae_diffusion_loss=loss_vae_diffusion_loss.detach(),
            ),
            dict(
                latents=latents,
                images_pred=images_pred,
            ),
        )

    def train_dis_step(self, batch):
        images = batch["image"].to(get_device(), memory_format=self.memory_format)
        self.add_batch_stat(images, "dis")

        # Pass through autoencoder.
        with torch.no_grad():
            images_pred = self.vae(images)[0]

        # Pass through discriminator.
        self.dis.requires_grad_(True)
        images = images.detach().requires_grad_(True)
        logits_real = self.dis(images)
        logits_fake = self.dis(images_pred)

        # Compute mse on reconstruction
        loss_dis = dis_loss(
            loss_type=self.config.loss.dis_type,
            logits_real=logits_real,
            logits_fake=logits_fake,
        )

        loss_dis = self.config.loss.gan_weight * loss_dis

        return dict(
            loss_dis=loss_dis,
        )
