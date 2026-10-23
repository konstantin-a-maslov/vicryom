import dataloader
from tqdm import tqdm


def run(cfg):
    train_data = dataloader.get_dataloader(
        cfg.manifest,
        subset="train",
        sample_size=cfg.patch_size,
        num_records=cfg.num_records,
        batch_size=cfg.batch_size,
        name="Train dataloader",
        worker_count=cfg.worker_count,
        worker_buffer_size=cfg.worker_buffer_size,
    )
    val_data = dataloader.get_dataloader(
        cfg.manifest,
        subset="val",
        sample_size=cfg.patch_size,
        batch_size=cfg.batch_size,
        name="Val dataloader",
        worker_count=cfg.worker_count,
        worker_buffer_size=cfg.worker_buffer_size,
    )

    steps_per_epoch = cfg.num_records // cfg.batch_size
    train_iter = iter(train_data)

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}")

        # Train loop
        with tqdm(total=steps_per_epoch, desc="") as pbar:
            for step in range(steps_per_epoch):
                pbar.set_description(f"{step + 1}/{steps_per_epoch}")
                images, masks = next(train_iter)
                # train_step(images, masks)
                pbar.update(1)

                if step % 1000 == 0:
                    plot(images, masks)

        # Val check
        print("Validation:", end=' ')

        i = 0
        for images, masks in val_data:
            # val_step(images, masks)
            if i % 500 == 0:
                plot(images, masks)
            i += 1
            pass

        print(".")

        break


# to remove later
import matplotlib.pyplot as plt


def plot(images, masks):
    batch_size = len(images)

    _, axs = plt.subplots(nrows=2, ncols=batch_size, figsize=(1 * batch_size, 3))

    for j, (image, mask) in enumerate(zip(images, masks)):
        axs[0][j].imshow(image[0], cmap="gray")
        axs[1][j].imshow(mask[0], cmap="cividis")

    for ax in axs.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.show()
