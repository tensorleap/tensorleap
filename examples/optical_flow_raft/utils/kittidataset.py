from utils.cloud_utils import _connect_to_gcs_and_return_bucket


class KITTIDataset:
    """KITTI optical flow dataset.
    List of adapted KITTI12 sizes: [(1, 375, 1242, 2), (1, 370, 1226, 2), (1, 376, 1241, 2), (1, 374, 1238, 2)]
    List of adapted KITTI15 sizes: [(1, 375, 1242, 2), (1, 370, 1224, 2), (1, 376, 1241, 2), (1, 374, 1238, 2)]
    """

    def __init__(self,
                 ds_root,
                 opt_type: str,
                 bucket_name: str,
                 val_split: float = 0.25,
                 random_seed: int = 0,
                 data_subset: str = "stereo"
                 ):
        """Initialize the KITTIDataset object
        Args:
            mode: Possible options: 'train_noval', 'val', 'train_with_val' or 'test'
            ds_root: Path to the root of the dataset
            options: see base class documentation
        Flow stats:
            KITTI2012: training flow mag min=0.0, avg=6.736172669242419, max=232.20108032226562 (194 flows)
            KITTI2015: raining flow mag min=0.0, avg=4.7107220490319, max=256.4881896972656 (200 flows)
        """
        assert (opt_type in ['noc', 'occ'])
        self.ds_root = ds_root
        self.opt_type = opt_type
        self.val_split = val_split
        self.random_seed = random_seed
        self._set_folders(data_subset)
        self._build_ID_sets(bucket_name)

    def _set_folders(self, dataset_subset="stereo"):  # noc, occ
        """Set the train, val, test, label and prediction label folders.
        """
        if dataset_subset == "stereo":
            image_folder = "colored_0"
        else:
            image_folder = "image_2"
        self._train_input_dir = self.ds_root + f'/training/{image_folder}'
        self._val_dir = self._train_input_dir
        self._test_dir = self.ds_root + f'/testing/{image_folder}'

        self._train_gt_dir = self.ds_root + '/training/flow_' + self.opt_type
        self._val_gt_dir = self._train_gt_dir

    def _build_ID_sets(self, bucket_name):
        """Build the list of samples and their IDs, split them in the proper datasets.
        Called by the base class on init.
        Each ID is a tuple.
        For the training/val datasets, they look like ('000065_10.png', '000065_11.png', '000065_10.png')
         -> gt flows are stored as 48-bit PNGs
        For the test dataset, they look like ('000000_10.png', '00000_11.png', '000000_10.flo')
        """
        # Search the train folder for the samples, create string IDs for them

        bucket = _connect_to_gcs_and_return_bucket(bucket_name)
        input_frames = [obj.name for obj in bucket.list_blobs(prefix=self._train_input_dir)]
        gt_frames = [obj.name for obj in bucket.list_blobs(prefix=self._train_gt_dir)]
        self.train_IDs = [(input_frames[idx1], input_frames[idx1 + 1], gt_frames[idx2]) for idx1, idx2
                    in zip(range(0, len(input_frames) - 1, 2), range(len(gt_frames)))]


def get_kitti_data(bucket_name, data_subset='stereo') -> KITTIDataset:
    if data_subset == 'stereo':
        dataset_path = 'KITTI/data_stereo_flow'
    elif data_subset == 'scene':
        dataset_path = 'KITTI/data_scene_flow'
    kitti_ds = KITTIDataset(ds_root=dataset_path, opt_type='noc', data_subset=data_subset, bucket_name=bucket_name)
    return kitti_ds
