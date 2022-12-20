import torch


def rdg_train_epoch(
    models,
    dataloader,
    optimizers,
    schedulers,
    criterions,
    specific_params,
    device,
    epoch,
    log_wandb,
):
    size = len(dataloader)
    pbar = tqdm(dataloader, total=size)
    for (real_img, similar_img, wrong_img, spec, spec_len) in pbar:
        real_img, similar_img, wrong_img, spec, spec_len = (
            real_img.to(device),
            similar_img.to(device),
            wrong_img.to(device),
            spec.to(device),
            spec_len.to(device),
        )

        for i in optimizers.keys():
            optimizers[i].zero_grad()

        bs = real_img.size(0)

        Z = torch.randn(bs, specific_params.latent_space_dim, device=device)
        A = models["sed"](spec, spec_len)

        fake_imgs, mu, logvar = models["gen"](Z, A)

        zero_labels = torch.zeros(bs, device=device, dtype=torch.long)
        one_labels = torch.ones(bs, device=device, dtype=torch.long)
        two_labels = torch.zeros(bs, device=device, dtype=torch.long) + 2

        # ---- Update D

        disc_loss = 0
        gen_loss = 0

        disc_labels = torch.ones(bs, device=device)
        for img_dim in specific_params.img_dims:
            disc_name = f"disc_{img_dim}"

            real_out = models[disc_name](real_img, mu.detach())
            wrong_out = models[disc_name](wrong_img, mu.detach())
            fake_out = models[disc_name](fake_imgs[img_dim], mu.detach())

            loss_real_cond = criterions["bce"](real_out["cond"], one_labels)
            loss_real_uncond = criterions["bce"](real_out["uncond"], one_labels)
            # ---
            loss_wrong_cond = criterions["bce"](wrong_out["cond"], zero_labels)
            loss_wrong_uncond = criterions["bce"](wrong_out["uncond"], one_labels)
            # ---
            loss_fake_cond = criterions["bce"](fake_out["cond"], zero_labels)
            loss_fake_uncond = criterions["bce"](fake_out["uncond"], zero_labels)

            loss_g = criterions["bce"](fake_out["cond"], one_labels)
            loss_g += criterions["bce"](fake_out["uncond"], one_labels)

            disc_loss += (
                loss_real_cond
                + loss_fake_uncond
                + loss_wrong_cond
                + loss_wrong_uncond
                + loss_fake_cond
                + loss_fake_uncond
            )

            gen_loss += loss_g

        disc_loss.backward()
        optimizers["disc"].step()
        schedulers["disc"].step()

        # ---- Update G and RS
        # --- RS

        real_feat = models["ied"](real_img)
        similar_feat = models["ied"](similar_img)
        fake_feat = models["ied"](fake_imgs[-1])
        wrong_feat = models["ied"](wrong_img)

        R1 = models["rs"](similar_feat, real_feat)
        R2 = models["rs"](wrong_feat, real_feat)
        R3 = models["rs"](real_feat, real_feat)
        R_GT_FI = models["rs"](fake_feat, real_feat)

        # loss here
        rs_loss = criterions["rs"](
            R1, R2, R3, R_GT_FI, zero_labels, one_labels, two_labels
        )

        # --- G
        gen_loss += criterions["kl"](mu, logvar) * specific_params.kl_loss_coef
        gen_loss.backward()
        optimizers["gen"].step()
        schedulers["gen"].step()

        if log_wandb:
            wandb.log({"train/G_loss": gen_loss.item()})
            wandb.log({"train/D_loss": disc_loss.item()})
            wandb.log({"train/epoch": epoch})
            wandb.log({"train/lr-OneCycleLR_G": scheduler["gen"].get_last_lr()[0]})
            wandb.log({"train/lr-OneCycleLR_D": scheduler["disc"].get_last_lr()[0]})

        pbar.set_description(
            f"[Epoch: {epoch}] G_Loss: {gen_loss.item():.2f} | D_Loss: {gen_loss.item():.2f}"
        )
