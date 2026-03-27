from utils.cropped_data_loader import get_data_loaders as private_loaders
from utils.public_data_loader import get_data_loaders as public_loaders


def get_data_loaders(public_path, private_path, batch_size=64):
    public_train_loader, public_val_loader, public_test_loader = public_loaders(
        data_dir=public_path,
        json_dir="pub_json",     
        batch_size=batch_size,
        use_mixup=False
    )
    
    private_train_loader, private_val_loader, private_test_loader = private_loaders(
        data_dir=private_path,
        json_dir="priv_json",
        batch_size=batch_size,
        use_mixup=False
    )

    print(f"Loaders ready.")
    print(f"Public: Train={len(public_train_loader)}, Val={len(public_val_loader)}")
    print(f"Private: Train={len(private_train_loader)}, Val={len(private_val_loader)}")

    return (public_train_loader, public_val_loader, public_test_loader,
            private_train_loader, private_val_loader, private_test_loader)