from .transforms import transforms
from torch.utils.data import DataLoader
from .mydataset import dataset as my_dataset
import torch


def data_loader(args, test_path=False, segmentation=False):

    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    input_size = (int(args.input_size), int(args.input_size))
    crop_size = (int(args.crop_size), int(args.crop_size))

    # transformation for training set
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  # 256
                                     transforms.RandomCrop(crop_size),  # 224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)])

    # transformation for test cls set
    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = [transforms.Resize(crop_size),
                           transforms.CenterCrop(crop_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean_vals, std_vals),]
    tsfm_clstest = transforms.Compose(func_transforms)

    # transformation for test loc set
    tsfm_loctest = transforms.Compose([transforms.Resize(crop_size),  # 224
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)])

    # training and test dataset & dataloader
    img_train = my_dataset(args.train_list, root_dir=args.img_dir, transform=tsfm_train, with_path=True, num_classes=200, datalist_file_root=args.train_root_list, datalist_file_parent=args.train_parent_list)
    img_clstest = my_dataset(args.test_list, root_dir=args.img_dir, transform=tsfm_clstest, with_path=test_path, num_classes=200)
    img_loctest = my_dataset(args.test_list, root_dir=args.img_dir, transform=tsfm_loctest, with_path=test_path, num_classes=200)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valcls_loader = DataLoader(img_clstest, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valloc_loader = DataLoader(img_loctest, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, valcls_loader, valloc_loader


