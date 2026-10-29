from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data.horse_zebra import UnpairedImageDataset
from models.cyclegan import ResnetGenerator


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, base_channels, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        current = base_channels
        for multiplier in (2, 4):
            following = base_channels * multiplier
            layers.extend([
                nn.Conv2d(current, following, 4, 2 if multiplier == 2 else 2, 1, bias=False),
                nn.InstanceNorm2d(following), nn.LeakyReLU(0.2, True),
            ])
            current = following
        layers.extend([
            nn.Conv2d(current, base_channels * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(base_channels * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class ImagePool:
    def __init__(self, capacity: int = 50) -> None:
        self.capacity = capacity
        self.images: list[torch.Tensor] = []

    def query(self, batch: torch.Tensor) -> torch.Tensor:
        results = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.capacity:
                self.images.append(image)
                results.append(image)
            elif random.random() > 0.5:
                index = random.randrange(self.capacity)
                old = self.images[index].clone()
                self.images[index] = image
                results.append(old)
            else:
                results.append(image)
        return torch.cat(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an unpaired Horse-to-Zebra and Zebra-to-Horse CycleGAN")
    parser.add_argument("--data", type=Path, default=Path("data/horse2zebra"))
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--decay-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or not 0 <= args.decay_epoch < args.epochs:
        parser.error("epochs must be positive and decay-epoch must be in [0, epochs)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UnpairedImageDataset(args.data, args.image_size, training=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)

    generator_ab = ResnetGenerator().to(device)
    generator_ba = ResnetGenerator().to(device)
    discriminator_a = PatchDiscriminator().to(device)
    discriminator_b = PatchDiscriminator().to(device)
    optim_g = torch.optim.Adam(list(generator_ab.parameters()) + list(generator_ba.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
    optim_da = torch.optim.Adam(discriminator_a.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_db = torch.optim.Adam(discriminator_b.parameters(), lr=args.lr, betas=(0.5, 0.999))
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    pool_a, pool_b = ImagePool(), ImagePool()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "losses.csv"
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "generator", "discriminator_a", "discriminator_b", "cycle", "identity"])
        for epoch in range(args.epochs):
            totals = torch.zeros(5, dtype=torch.float64)
            for batch in loader:
                real_a, real_b = batch["A"].to(device), batch["B"].to(device)
                optim_g.zero_grad(set_to_none=True)
                fake_b = generator_ab(real_a)
                fake_a = generator_ba(real_b)
                cycle_a = generator_ba(fake_b)
                cycle_b = generator_ab(fake_a)
                identity_a = generator_ba(real_a)
                identity_b = generator_ab(real_b)
                prediction_b = discriminator_b(fake_b)
                prediction_a = discriminator_a(fake_a)
                loss_gan_ab = mse(prediction_b, torch.ones_like(prediction_b))
                loss_gan_ba = mse(prediction_a, torch.ones_like(prediction_a))
                loss_cycle = 10.0 * (l1(cycle_a, real_a) + l1(cycle_b, real_b))
                loss_identity = 5.0 * (l1(identity_a, real_a) + l1(identity_b, real_b))
                loss_g = loss_gan_ab + loss_gan_ba + loss_cycle + loss_identity
                loss_g.backward()
                optim_g.step()

                optim_da.zero_grad(set_to_none=True)
                fake_a_pool = pool_a.query(fake_a)
                pred_real_a = discriminator_a(real_a)
                pred_fake_a = discriminator_a(fake_a_pool)
                loss_da = 0.5 * (mse(pred_real_a, torch.ones_like(pred_real_a)) +
                                 mse(pred_fake_a, torch.zeros_like(pred_fake_a)))
                loss_da.backward()
                optim_da.step()

                optim_db.zero_grad(set_to_none=True)
                fake_b_pool = pool_b.query(fake_b)
                pred_real_b = discriminator_b(real_b)
                pred_fake_b = discriminator_b(fake_b_pool)
                loss_db = 0.5 * (mse(pred_real_b, torch.ones_like(pred_real_b)) +
                                 mse(pred_fake_b, torch.zeros_like(pred_fake_b)))
                loss_db.backward()
                optim_db.step()
                totals += torch.tensor([loss_g.item(), loss_da.item(), loss_db.item(),
                                        loss_cycle.item(), loss_identity.item()], dtype=torch.float64)

            count = max(1, len(loader))
            means = (totals / count).tolist()
            writer.writerow([epoch + 1, *[f"{value:.6f}" for value in means]])
            stream.flush()
            print(f"Epoch {epoch + 1}/{args.epochs} | G {means[0]:.3f} | D_A {means[1]:.3f} | D_B {means[2]:.3f}")
            torch.save(generator_ab.cpu().state_dict(), args.output / "horse2zebra.pth")
            torch.save(generator_ba.cpu().state_dict(), args.output / "zebra2horse.pth")
            if (epoch + 1) % 5 == 0 or epoch == 0:
                sample_dir = args.output.parent / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)
                save_image(fake_b[:1].detach().cpu(), sample_dir / f"epoch-{epoch + 1:03d}-horse2zebra.png", normalize=True)
            generator_ab.to(device)
            generator_ba.to(device)
            if epoch + 1 > args.decay_epoch:
                factor = max(0.0, 1 - (epoch + 1 - args.decay_epoch) / (args.epochs - args.decay_epoch))
                for optimizer in (optim_g, optim_da, optim_db):
                    for group in optimizer.param_groups:
                        group["lr"] = args.lr * factor

    print(f"Training finished. Weights and loss history are in {args.output.resolve()}")


if __name__ == "__main__":
    main()
