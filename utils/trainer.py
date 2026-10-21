from models import (
    Generator,
    Discriminator
)

import os

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import torchutils

from .hf_utils import (
    ImageDataset
)
from .gen_utils import(
    weights_init
)

def get_argparser():
    parser = torchutils.ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["edges2shoes"],
        help="Dataset to train on. Must be one of: `edges2shoes`",
    )
    parser.add_argument(
        "--generator_in_channels",
        type=int,
        nargs="+",
        help="Input channels for the generator.",
    )
    parser.add_argument(
        "--num_residual_blocks",
        type=int,
        default=9,
        help="Number of residual blocks to use in the generator.",
    )
    parser.add_argument(
        "--discriminator_in_channels",
        type=int,
        nargs="+",
        help="Input channels for the discriminator.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads to use in data loading (default: 4)"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Whether to pin memory for data loading (default: True)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train for (default: 200)",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=256,
        help="Height of the input images (default: 256)",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="Width of the input images (default: 256)",
    )
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=0.0002,
        help="Learning rate for the generator (default: 0.0002).",
    )
    parser.add_argument(
        "--generator_adam_beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5).",
    )
    parser.add_argument(
        "--generator_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=0.0002,
        help="Learning rate for the discriminator (default: 0.0002).",
    )
    parser.add_argument(
        "--discriminator_adam_beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5).",
    )
    parser.add_argument(
        "--discriminator_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--pixel_loss_weight",
        type=float,
        default=100.0,
        help="Pixel loss weight (default: 100.0).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(".", "checkpoints"),
        help="Directory to save checkpoints (default: ./checkpoints).",
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=200,
        help="Interval (in epochs) at which to save model checkpoints (default: 200).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(".", "output"),
        help="Directory to save intermediate output (default: ./output).",
    )
    parser.add_argument(
        "--save_output_interval",
        type=int,
        default=50,
        help="Number of epochs between saving intermediate output (default: 50).",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=9,
        help="Number of rows in output grid (default: 9).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    return parser

def main():
    args = get_argparser().parse_args()
    pad_length = len(str(args.num_epochs))

    torchutils.set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Training on {args.dataset}")
    if args.dataset == "edges2shoes":
        dataset = load_dataset("luethan2025/edges2shoes", split="train")
        transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImageDataset(
            hf_dataset=dataset,
            transform=transform
        )
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            drop_last=True,
        )
        dataset = "edges2shoes"
        img_size = (3, args.image_height, args.image_width)

    G = Generator(
        img_size=img_size,
        in_channels=args.generator_in_channels,
    )
    G.apply(weights_init)
    D = Discriminator(
        img_size=img_size,
        in_channels=args.discriminator_in_channels,
    )
    D.apply(weights_init)

    G = G.to(device)
    D = D.to(device)

    G_optimizer = optim.Adam(G.parameters(), lr=args.generator_lr, betas=(args.generator_adam_beta1, args.generator_adam_beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=args.discriminator_lr, betas=(args.discriminator_adam_beta1, args.discriminator_adam_beta2))

    adversarial_criterion = torch.nn.MSELoss()
    pixel_criterion = torch.nn.L1Loss()

    dtype = torchutils.get_torch_dtype(args.mixed_precision)
    use_amp = (device.type == "cuda") and (dtype != torch.float32)
    use_scaler = (device.type == "cuda") and (dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    if not (os.path.exists(args.checkpoint_dir) and os.path.isdir(args.checkpoint_dir)):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    output_dir = os.path.join(args.output_dir, args.dataset, "epoch")
    if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset)
    if not (os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir)):
        os.makedirs(checkpoint_dir, exist_ok=True)

    patch_size = (
        1,
        img_size[1] // 2 ** 4,
        img_size[2] // 2 ** 4,
    )

    indices = torch.randperm(len(dataloader.dataset))[:args.nrow ** 2].tolist()
    test_images = torch.stack(
        [dataloader.dataset[i][0] for i in indices]
    ).to(device)

    for epoch in range(args.num_epochs):
        running_G_loss = 0.0
        running_D_loss = 0.0
        with tqdm(dataloader, desc="Training", unit="batch") as pbar:
            for idx, (realA, realB) in enumerate(pbar):
                realA = realA.to(device)
                realB = realB.to(device)

                real_labels = torch.ones(
                    (realA.size(0), *patch_size),
                    device=realA.device,
                )
                fake_labels = torch.zeros(
                    (realA.size(0), *patch_size),
                    device=realA.device,
                )

                # === Train Generator ===
                G_optimizer.zero_grad()

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    fakeB = G(realA)
                    adversarial_loss = adversarial_criterion(D(fakeB, realA), real_labels)
                    pixel_loss = pixel_criterion(fakeB, realB)

                    G_loss = adversarial_loss + \
                            args.pixel_loss_weight * pixel_loss

                scaler.scale(G_loss).backward()
                scaler.step(G_optimizer)

                running_G_loss += G_loss.item()

                # === Train Discriminator A ===
                D_optimizer.zero_grad()

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    real_loss = adversarial_criterion(D(realB, realA), real_labels)
                    fake_loss = adversarial_criterion(D(fakeB.detach(), realA), fake_labels)
                    D_loss = (real_loss + fake_loss) / 2

                scaler.scale(D_loss).backward()
                scaler.step(D_optimizer)

                running_D_loss += D_loss.item()

                scaler.update()

                pbar.set_postfix({
                    "G_loss": f"{G_loss.item():.3f}",
                    "D_loss": f"{D_loss.item():.3f}",
                })

        average_G_loss = running_G_loss / len(dataloader)
        average_D_loss = running_D_loss / len(dataloader)

        print(f"Epoch: {epoch + 1}/{args.num_epochs}")
        print(f"G Loss: {average_G_loss:.5f}, D Loss: {average_D_loss:.5f}")

        if (epoch + 1) % args.save_output_interval == 0:
            print("Saving fake images")
            G.eval()
            with torch.no_grad():
                fake_images = G(test_images)
            grid = torchvision.utils.make_grid(
                fake_images,
                nrow=args.nrow,
                normalize=True,
                value_range=(-1, 1)
            )
            torchvision.utils.save_image(
                grid,
                os.path.join(output_dir, f"{epoch + 1:0{pad_length}d}.png")
            )
            G.train()

        if (epoch + 1) % args.save_checkpoint_interval == 0:
            print("Saving model checkpoints")
            checkpoint = {
                "dataset": args.dataset,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "generator_in_channels": args.generator_in_channels,
                "discriminator_in_channels": args.discriminator_in_channels,
                "generator_lr": args.generator_lr,
                "discriminator_lr": args.discriminator_lr,
                "G_optimizer": G_optimizer.state_dict(),
                "D_optimizer": D_optimizer.state_dict(),
                "image_height": args.image_height,
                "image_width": args.image_width,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"{args.dataset}_checkpoint_{epoch + 1:0{pad_length}d}.pth"))
            generator = {
                "dataset": args.dataset,
                "G": G.state_dict(),
                "generator_in_channels": args.generator_in_channels,
                "img_size": img_size,
            }
            torch.save(generator, os.path.join(checkpoint_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.pth"))

    print("Saving final model checkpoints")
    checkpoint = {
        "dataset": args.dataset,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "generator_in_channels": args.generator_in_channels,
        "discriminator_in_channels": args.discriminator_in_channels,
        "generator_lr": args.generator_lr,
        "discriminator_lr": args.discriminator_lr,
        "G_optimizer": G_optimizer.state_dict(),
        "D_optimizer": D_optimizer.state_dict(),
        "image_height": args.image_height,
        "image_width": args.image_width,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"{args.dataset}_checkpoint_{epoch + 1:0{pad_length}d}.pth"))
    generator = {
        "dataset": args.dataset,
        "G": G.state_dict(),
        "generator_in_channels": args.generator_in_channels,
        "img_size": img_size,
    }
    torch.save(generator, os.path.join(checkpoint_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.pth"))

if __name__ == "__main__":
    main()
