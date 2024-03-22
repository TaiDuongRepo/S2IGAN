from data.dataset import SENDataset


def main():
    dataset = SENDataset(
        r"/kaggle/input/s2igan-oxford/s2igan_oxford/train_flower_en2vi.json",
        r"/kaggle/input/s2igan-oxford/s2igan_oxford/image_oxford/image_oxford",
        r"/kaggle/input/s2igan-oxford/s2igan_oxford/oxford_audio/oxford_audio",
    )


if __name__ == "__main__":
    main()
