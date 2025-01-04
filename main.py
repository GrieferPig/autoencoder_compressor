"""
Command-line interface supporting commands like:
  main.py train ae 2,64,16 --epoch 1500 --plot
  main.py train denoise gaussian --img-set-size 1000
  main.py train denoise ae 2,64,16 --epoch 100 --plot
  main.py inference ae 2,64,16 --indices 1,2,3,4
  main.py inference denoise gaussian --use-ae 2,256,64
  main.py inference denoise ae 2,64,16 --use-ae 2,256,64 --indices 2,3,4,5
  main.py inference denoise oidn --use-ae 2,256,64
"""

import argparse
import sys
from config import *
from autoenc import (
    train_ae_model,
    inference_ae_model,
    plot_ae_model,
)
from denoise import (
    train_denoise_model,
    inference_denoise_model,
    plot_denoise_results,
)
from utils import (
    get_model_size,
    load_autoenc_model,
    load_denoise_model,
)
from AEDataset import AEDataset, init_ae_dataset
from DenoisingDataset import *
from torch.utils.data import DataLoader
import os
from datasets import load_dataset


def parse_enc_config(s):
    """Parse a string of comma-separated integers into a list, e.g. '2,64,16' -> [2, 64, 16]."""
    if not s:
        return None
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Train/Inference for AE or Denoise with optional configs.",
    )
    parser.add_argument(
        "subcommand", choices=["train", "inference", "getsize"], help="Subcommand"
    )
    parser.add_argument(
        "model_type", choices=["ae", "denoise"], help="Model type: ae or denoise"
    )
    parser.add_argument(
        "extra",
        nargs="?",
        default=None,
        help=(
            "[AE] pass enc config like '2,64,16', "
            "[Denoise] pass method like 'gaussian', 'ae', or 'oidn'. "
            "If method='ae', the next arg is enc config."
        ),
    )
    parser.add_argument(
        "enc_config",
        nargs="?",
        default=None,
        help="Optional enc config if denoise method=ae.",
    )
    parser.add_argument(
        "--epoch", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument(
        "--img-set-size",
        type=int,
        default=None,
        help="Image set size (override config default)",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="0,1,2,3",
        help="Comma-separated indices for inference",
    )
    parser.add_argument(
        "--use-ae",
        type=str,
        default=None,
        help="Comma-separated enc config to use for AE-based denoising",
    )
    parser.add_argument(
        "--continue",
        type=str,
        default=None,
        help="Continue training from given epoch or final checkpoint",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Train AE model until convergence (SSIM improvement < 0.01 in last 50 epochs)",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset")
    parser.add_argument(
        "--ssim", action="store_true", help="Use SSIM loss for training"
    )
    parser.add_argument(
        "--decoder", action="store_true", help="Get decoder size instead of full model"
    )
    parser.add_argument("--img-source", type=str, default=None, help="Image source")
    parser.add_argument("--generic", action="store_true", help="Load generic model")

    args = parser.parse_args()
    enc_config_str = None
    method = None

    if args.model_type == "ae":
        # Expect something like main.py train ae 2,64,16 ...
        # or main.py inference ae 2,64,16 ...
        enc_config_str = args.extra
    else:
        # model_type == "denoise"
        # e.g. main.py train denoise gaussian
        #      main.py train denoise ae 2,64,16
        #      main.py inference denoise gaussian
        #      main.py inference denoise ae 2,64,16
        #      main.py inference denoise oidn
        method = args.extra
        if method == "ae":
            enc_config_str = args.enc_config

    enc_config = parse_enc_config(enc_config_str)
    use_ae_config = parse_enc_config(args.use_ae)
    epochs = args.epoch

    # Defaults from config if not specified
    if args.model_type == "ae" and epochs is None:
        epochs = EPOCHS_BASE_AE
    elif args.model_type == "denoise" and epochs is None:
        epochs = EPOCHS_DENOISE

    if args.subcommand == "train":
        if args.model_type == "ae":
            if enc_config == None:
                print("Missing enc config for AE-based denoising.")
                sys.exit(1)
            enc_layers, img_set_size, latent_dim = enc_config
            img_set_size = (
                img_set_size if args.img_set_size is None else args.img_set_size
            )
            print(
                f"Training AE with ENC_LAYERS={enc_layers}, IMG_SET_SIZE={img_set_size}, LATENT_DIM={latent_dim}, "
                f"epochs={epochs}, plot={args.plot}"
            )
            if args.convergence:
                print("Actually, forget about the epochs, we train until convergence!")

            # Initialize base dataset
            if args.img_source:
                base_dataset = args.img_source
            else:
                base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)

            # Initialize dataset and dataloader
            dataset, dataloader = init_ae_dataset(
                base_dataset, img_set_size, shuffle=args.shuffle
            )

            # Train the model
            model, avg_loss = train_ae_model(
                dataloader=dataloader,
                ENC_LAYERS=enc_layers,
                IMG_SET_SIZE=img_set_size,
                LATENT_DIM=latent_dim,
                custom_dataset=dataset,
                epochs=epochs,
                till_convergence=args.convergence,
                use_ssim=args.ssim,
            )

            print(f"AE training completed. Final avg_loss: {avg_loss:.4f}")

            if args.plot:
                # Perform inference for plotting
                mean_similarity, avg_psnr, avg_ssim = inference_ae_model(
                    model, dataloader, num_examples=100
                )
                plot_ae_model(
                    model=model,
                    custom_dataset=dataset,
                    ENC_LAYERS=enc_layers,
                    IMG_SET_SIZE=img_set_size,
                    LATENT_DIM=latent_dim,
                    avg_loss=avg_loss,
                    mean_similarity=mean_similarity,
                    avg_psnr=avg_psnr,
                    avg_ssim=avg_ssim,
                )
        else:  # denoise
            if method == "gaussian":
                print(
                    f"Training Denoise with method='gaussian', epochs={epochs}, img_set_size={args.img_set_size}"
                )
                img_set_size = args.img_set_size if args.img_set_size else 1000
                dataset = DenoisingDatasetGaussian(img_set_size=img_set_size)
            elif method == "ae":
                enc_layers, img_set_size, latent_dim = enc_config
                img_set_size = (
                    img_set_size if args.img_set_size is None else args.img_set_size
                )
                print(
                    f"Training Denoise with method='ae', ENC_LAYERS={enc_layers}, IMG_SET_SIZE={img_set_size}, "
                    f"LATENT_DIM={latent_dim}, epochs={epochs}"
                )
                # Load the trained AE model
                source_ae_model, (model_epochs, model_loss, model_indices) = (
                    load_autoenc_model(
                        enc_layers=enc_layers,
                        img_set_size=img_set_size,
                        latent_dim=latent_dim,
                        load_optimizer=False,
                    )
                )
                print(
                    f"Loaded AE model with ENC_LAYERS={enc_layers}, IMG_SET_SIZE={img_set_size}, LATENT_DIM={latent_dim}, "
                    f"epochs={model_epochs}, loss={model_loss:.4f}, indices={model_indices}"
                )
                # Initialize base dataset
                base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)
                dataset = DenoisingDataset(base_dataset, source_ae_model, model_indices)
            elif method == "oidn":
                print(f"Training Denoise with method='oidn', epochs={epochs}")
                raise NotImplementedError("OIDN training not implemented yet.")
            else:
                print(
                    "Invalid denoise method. Choose from 'gaussian', 'ae', or 'oidn'."
                )
                sys.exit(1)

            dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE_DENOISE_DATA, shuffle=args.shuffle
            )

            # Train the denoise model
            model, training_losses = train_denoise_model(
                dataloader=dataloader,
                gaussian_noise_model=(method == "gaussian"),
                enc_layers=enc_config[0] if method == "ae" else None,
                img_set_size=enc_config[1] if method == "ae" else None,
                latent_dim=enc_config[2] if method == "ae" else None,
                epochs=epochs,
            )

            print(f"Denoise training completed. Final loss: {training_losses[-1]:.4f}")

            if args.plot:
                plot_denoise_results(
                    model=model,
                    dataset=dataset,
                    gaussian_noise_model=(method == "gaussian"),
                    enc_layers=enc_config[0] if method == "ae" else None,
                    img_set_size=enc_config[1] if method == "ae" else None,
                    latent_dim=enc_config[2] if method == "ae" else None,
                )

    elif args.subcommand == "inference":
        if args.model_type == "ae":
            enc_layers, img_set_size, latent_dim = enc_config
            print(
                f"Inference AE with ENC_LAYERS={enc_layers}, IMG_SET_SIZE={img_set_size}, LATENT_DIM={latent_dim}, "
                f"indices={args.indices}"
            )

            # Load the trained AE model
            model = load_autoenc_model(
                enc_layers=enc_layers,
                img_set_size=img_set_size,
                latent_dim=latent_dim,
                load_optimizer=False,
            )

            # Initialize dataset and dataloader
            if args.img_source:
                base_dataset = load_dataset(args.img_source, split=DATASET_SPLIT)
            else:
                base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)

            dataset, dataloader = init_ae_dataset(
                base_dataset, img_set_size, indices=args.indices, shuffle=False
            )

            # Perform inference
            mean_similarity, avg_psnr, avg_ssim = inference_ae_model(
                model, dataloader, num_examples=100
            )
            print(
                f"AE Inference Results - Mean Cosine Similarity: {mean_similarity:.4f}, "
                f"Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}"
            )

            if args.plot:
                plot_ae_model(
                    model=model,
                    custom_dataset=dataset,
                    ENC_LAYERS=enc_layers,
                    IMG_SET_SIZE=img_set_size,
                    LATENT_DIM=latent_dim,
                    avg_loss=0,  # Replace with actual loss if available
                    mean_similarity=mean_similarity,
                    avg_psnr=avg_psnr,
                    avg_ssim=avg_ssim,
                )

        else:  # denoise
            method = method  # 'gaussian', 'ae', or 'oidn'
            print(
                f"Inference Denoise with method='{method}', use_ae_config={use_ae_config}, indices={args.indices}"
            )

            if method == "gaussian":
                model = load_denoise_model(
                    gaussian_noise_model=True,
                    load_optimizer=False,
                )
                dataset = DenoisingDatasetGaussian()
            elif method == "ae":
                indices = args.indices.split(",")

                enc_layers, img_set_size, latent_dim = enc_config
                if use_ae_config is None:
                    # use the same config as the AE model
                    use_ae_config = enc_config
                use_enc_layers, use_img_set_size, use_latent_dim = use_ae_config
                # load the trained AE model
                source_ae_model, (_, _, ae_indices) = load_autoenc_model(
                    enc_layers=enc_layers,
                    img_set_size=img_set_size,
                    latent_dim=latent_dim,
                    load_optimizer=False,
                )
                # Use indices to index ae_indices
                indices = [ae_indices[int(i)] for i in indices]
                model = load_denoise_model(
                    gaussian_noise_model=False,
                    generic_model=args.generic,
                    enc_layers=use_enc_layers,
                    img_set_size=use_img_set_size,
                    latent_dim=use_latent_dim,
                    load_optimizer=False,
                )

                if args.img_source:
                    base_dataset = load_dataset(args.img_source, split=DATASET_SPLIT)
                else:
                    base_dataset = load_dataset(DATASET_REPO, split=DATASET_SPLIT)
                dataset = DenoisingDataset(
                    base_dataset,
                    source_ae_model,
                    indices=indices,
                )
            elif method == "oidn":
                raise NotImplementedError("OIDN inference not implemented yet.")
            else:
                print(
                    "Invalid denoise method. Choose from 'gaussian', 'ae', or 'oidn'."
                )
                sys.exit(1)

            dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE_DENOISE_DATA, shuffle=False
            )

            # Perform inference
            mse, psnr = inference_denoise_model(
                model, dataloader, num_examples=NUM_SAMPLES_DENOISE
            )
            print(f"Denoise Inference Results - MSE: {mse:.4f}, PSNR: {psnr:.4f}")

            if args.plot:
                plot_denoise_results(
                    model=model,
                    dataset=dataset,
                    gaussian_noise_model=(method == "gaussian"),
                    enc_layers=enc_config[0] if method == "ae" else None,
                    img_set_size=enc_config[1] if method == "ae" else None,
                    latent_dim=enc_config[2] if method == "ae" else None,
                )

    elif args.subcommand == "getsize":
        # get model size on disk
        if args.model_type == "ae":
            enc_layers, img_set_size, latent_dim = enc_config
            model, _ = load_autoenc_model(
                enc_layers=enc_layers,
                img_set_size=img_set_size,
                latent_dim=latent_dim,
                load_optimizer=False,
            )
            if args.decoder:
                model = model.decoder
            model_size = get_model_size(model)
        elif args.model_type == "denoise":
            if method == "gaussian":
                model = load_denoise_model(
                    gaussian_noise_model=True, load_optimizer=False
                )
            elif method == "ae":
                enc_layers, img_set_size, latent_dim = enc_config
                model = load_denoise_model(
                    gaussian_noise_model=False,
                    enc_layers=enc_layers,
                    img_set_size=img_set_size,
                    latent_dim=latent_dim,
                    load_optimizer=False,
                )
            elif method == "oidn":
                raise NotImplementedError("OIDN inference not implemented yet.")
            else:
                print(
                    "Invalid denoise method. Choose from 'gaussian', 'ae', or 'oidn'."
                )
                sys.exit(1)

        print(f"AE model size: {model_size:.4f} MB")


if __name__ == "__main__":
    main()
