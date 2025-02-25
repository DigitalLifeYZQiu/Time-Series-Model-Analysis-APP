import os

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    UCRAnomalyloader, CIDatasetBenchmark, CIAutoRegressionDatasetBenchmark,AluminaMSDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'UCRA': UCRAnomalyloader,
    'alumina': AluminaMSDataset,
    'Gas': Dataset_Custom,
}


def target_data_provider(args, target_root_path, target_data_path, target_data, flag='test'):
    temp_root_path = args.root_path
    temp_data_path = args.data_path
    temp_data = args.data
    args.root_path = target_root_path
    args.data_path = target_data_path
    args.data = target_data
    target_data_set, target_data_loader = data_provider(args, flag)
    args.root_path = temp_root_path
    args.data_path = temp_data_path
    args.data = temp_data
    return target_data_set, target_data_loader


def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    Data = data_dict[args.data]
    # import pdb
    # pdb.set_trace()
    if args.task_name == 'large_finetune':
        if args.roll:
            if 'alumina' in args.data:
                data_set = AluminaMSDataset(
                    root_path=os.path.join(args.root_path, args.data_path),
                    feature_cols= args.feature_cols,
                    target_cols=args.target_cols,
                    flag=flag,
                    input_len=args.seq_len-args.pred_len,
                    label_len=args.label_len,
                    pred_len=args.output_len if flag == 'test' else args.pred_len,
                    data_type=args.data,
                    scale=True,
                    timeenc=timeenc,
                    freq=args.freq,
                    stride=args.stride,
                    subset_ratio=args.subset_ratio,
                    subset_rand_ratio=args.subset_rand_ratio,
                    use_post_data=args.use_post_data,
                    train_offset=args.train_offset,
                )
            else:
                data_set = CIAutoRegressionDatasetBenchmark(
                    root_path=os.path.join(args.root_path, args.data_path),
                    flag=flag,
                    input_len=args.seq_len,
                    label_len=args.label_len,
                    pred_len=args.output_len if flag == 'test' else args.pred_len,
                    data_type=args.data,
                    scale=True,
                    timeenc=timeenc,
                    freq=args.freq,
                    stride=args.stride,
                    subset_ratio=args.subset_ratio,
                    subset_rand_ratio=args.subset_rand_ratio,
                    use_post_data=args.use_post_data,
                    train_offset=args.train_offset,
                )
        else:
            data_set = CIDatasetBenchmark(
                root_path=os.path.join(args.root_path, args.data_path),
                flag=flag,
                input_len=args.seq_len,
                pred_len=args.pred_len,
                data_type=args.data,
                scale=True,
                timeenc=timeenc,
                freq=args.freq,
                stride=args.stride,
                subset_ratio=args.subset_ratio,
                subset_rand_ratio=args.subset_rand_ratio,
                use_post_data=args.use_post_data,
                train_offset=args.train_offset,
            )
        print(flag, len(data_set))
        if args.use_multi_gpu:
            train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(data_set,
                                     batch_size=args.batch_size,
                                     sampler=train_datasampler,
                                     num_workers=args.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True,
                                     drop_last=False,
                                     )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False)
        return data_set, data_loader
    elif args.task_name == 'ucr_anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
