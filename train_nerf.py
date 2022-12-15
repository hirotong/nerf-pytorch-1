import argparse
import glob
import os
import time
import imageio

import numpy as np
import cv2
import torch
import torchvision
import yaml
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from PIL import Image

from nerf import (
    CfgNode,
    get_embedding_function,
    get_ray_bundle,
    img2mse,
    load_blender_data,
    load_llff_data,
    meshgrid_xy,
    models,
    mse2psnr,
    run_one_iter_of_nerf,
)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to (.yml) config file.")
    parser.add_argument(
        "--load-checkpoint", type=str, default="", help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--case", type=str, required=True, help="data case name"
    )
    parser.add_argument(
        "--prob_sampling", action="store_true", help="wheter use probability based sampling strategy"
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        conf_text = f.read()
        conf_text = conf_text.replace("CASE_NAME", configargs.case)
    cfg_dict = yaml.load(conf_text, Loader=yaml.SafeLoader)
    cfg = CfgNode(cfg_dict)

    cfg.dataset.basedir = os.path.expanduser(cfg.dataset.basedir)
    cfg.nerf.train.prob_sampling = configargs.prob_sampling
    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "val", "*.data"))
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf = None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            images, poses, render_poses, hwf, i_split, probs = load_blender_data(
                cfg.dataset.basedir, half_res=cfg.dataset.half_res, testskip=cfg.dataset.testskip
            )
            i_train, i_val, i_test = i_split
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        elif cfg.dataset.type.lower() == "llff":
            images, poses, bds, render_poses, i_test, probs = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
            i_val = i_test
            i_train = np.array([i for i in np.arange(images.shape[0]) if (i not in i_test and i not in i_val)])
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            images = torch.from_numpy(images)
            poses = torch.from_numpy(poses)
            probs = torch.from_numpy(probs)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(trainable_parameters, lr=cfg.optimizer.lr)

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id + "_prob_sampling" if cfg.nerf.train.prob_sampling else '')
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # # TODO: Prepare raybatch tensor if batching random rays
    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.train()
        if model_fine:
            model_coarse.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(ray_origins.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False,)
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            rgb_coarse, _, acc_coarse, rgb_fine, _, acc_fine = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
        else:
            img_idx = np.random.choice(i_train)
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            prob_target = probs[img_idx].numpy().ravel()
            prob_target = prob_target / prob_target.sum()
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            coords = torch.stack(meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)), dim=-1,)
            coords = coords.reshape((-1, 2))
            if hasattr(cfg.nerf.train, "prob_sampling") and cfg.nerf.train.prob_sampling:
                select_inds = np.random.choice(
                    coords.shape[0], size=(cfg.nerf.train.num_random_rays), p=prob_target, replace=False
                )
            else:
                select_inds = np.random.choice(coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False)
            select_inds = coords[select_inds]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

            then = time.time()
            rgb_coarse, _, acc_coarse, rgb_fine, _, acc_fine = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
            target_ray_values = target_s

        if hasattr(cfg, 'loss') and cfg.loss.type == "Entropy":
            prob_coarse = torch.stack([torch.clip(1 - acc_coarse, 1e-8), acc_coarse], dim=-1)
            coarse_loss = torch.nn.functional.nll_loss(torch.log(prob_coarse), target_ray_values[..., -1].long())
            mse_coarse = torch.mean(
                (acc_coarse - target_ray_values[..., -1]) * (acc_coarse - target_ray_values[..., -1])
            )

            fine_loss = None

            if acc_fine is not None:
                prob_fine = torch.stack([torch.clip(1 - acc_coarse, 1e-8), acc_fine], dim=-1)
                fine_loss = torch.nn.functional.nll_loss(torch.log(prob_fine), target_ray_values[..., -1].long())
                mse_fine = torch.mean((acc_fine - target_ray_values[..., -1]) * (acc_fine - target_ray_values[..., -1]))

        else:
            coarse_loss = torch.nn.functional.mse_loss(rgb_coarse, target_ray_values[..., :3])
            fine_loss = None
            if rgb_fine is not None:
                fine_loss = torch.nn.functional.mse_loss(rgb_fine, target_ray_values[..., :3])
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = 0.0
        # if fine_loss is not None:
        #     loss = fine_loss
        # else:
        #     loss = coarse_loss
        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        loss.backward()
        if hasattr(cfg, 'loss') and cfg.loss.type == "Entropy":
            mse = mse_coarse  # + (mse_fine if fine_loss is not None else 0.0)
            psnr = mse2psnr(mse.item())
        else:
            psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (cfg.scheduler.lr_decay_factor ** (i / num_decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write("[TRAIN] Iter: " + str(i) + " Loss: " + str(loss.item()) + " PSNR: " + str(psnr))
        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if i % cfg.experiment.validate_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, acc_coarse, rgb_fine, _, acc_fine = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
                    rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                loss, fine_loss = 0.0, 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                    loss = fine_loss
                else:
                    loss = coarse_loss
                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validation/psnr", psnr, i)
                writer.add_image("validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i)
                writer.add_image("validataion/acc_coarse", visualize_depth(acc_coarse), i)

                writer.add_image("validataion/disp_coarse", visualize_depth(disp_coarse), i)

                if rgb_fine is not None:
                    writer.add_image("validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i)
                    writer.add_image("validation/acc_fine", visualize_depth(acc_fine), i)
                    writer.add_image("validation/disp_fine", visualize_depth(disp_fine), i)
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                writer.add_image(
                    "validation/img_target", cast_to_image(target_ray_values[..., :3]), i,
                )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )
        
        # Test
        if (i % cfg.experiment.test_every == 0 and i > 0) or i == cfg.experiment.train_iters - 1:
            tqdm.write("[TEST] =======> Iter: " + str(i))
            if cfg.experiment.save_image:
                savedir = os.path.join(logdir, f"test_{i:0>6d}")
                os.makedirs(savedir, exist_ok=True)
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                # TODO fulfill this part
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, acc_coarse, rgb_fine, _, acc_fine = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    psnr = []
                    for idx, img_idx in enumerate(i_test):
                        img_target = images[img_idx].to(device)
                        pose_target = poses[img_idx, :3, :4].to(device)
                        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
                        rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine = run_one_iter_of_nerf(
                            H,
                            W,
                            focal,
                            model_coarse,
                            model_fine,
                            ray_origins,
                            ray_directions,
                            cfg,
                            mode="validation",
                            encode_position_fn=encode_position_fn,
                            encode_direction_fn=encode_direction_fn,
                        )
                        target_ray_values = img_target
                        coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                        loss, fine_loss = 0.0, 0.0
                        if rgb_fine is not None:
                            fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                            loss = fine_loss
                        else:
                            loss = coarse_loss
                        # loss = coarse_loss + fine_loss
                        psnr.append(mse2psnr(loss.item()))
                        
                        # save test images
                        if cfg.experiment.save_image:
                            savefile = os.path.join(savedir, f"rgb_{idx:04d}.png")
                            imageio.imwrite(savefile, np.moveaxis(cast_to_image(rgb_fine[..., :3]), 0, -1))
                            savefile = os.path.join(savedir, f"disp_{idx:04d}.png")
                            imageio.imwrite(savefile, (255 * np.array(T.ToPILImage()(visualize_depth(disp_fine)))).astype(np.uint8))
                            savefile = os.path.join(savedir, f"gt_{idx:04d}.png")
                            imageio.imwrite(savefile, np.moveaxis(cast_to_image(target_ray_values[..., :3]), 0, -1))
                        
                        
                        
                    psnr = np.array(psnr).mean()
                    writer.add_scalar("validation/loss", loss.item(), i)
                    writer.add_scalar("test/coarse_loss", coarse_loss.item(), i)
                    writer.add_scalar("test/psnr", psnr, i)

                    if rgb_fine is not None:
                        writer.add_scalar("test/fine_loss", fine_loss.item(), i)
                    tqdm.write(
                        "test loss: "
                        + str(loss.item())
                        + " Test PSNR: "
                        + str(psnr)
                        + " Time: "
                        + str(time.time() - start)
                    )
        

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None if not model_fine else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            torch.save(
                checkpoint_dict, os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)
    mi = np.min(x)
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)
    return x_


if __name__ == "__main__":
    main()
