# Feature format

Each sample is stored as `<sample_id>.pt` and must contain exactly these tensors:

- `rgb`: float tensor `[T, 2048]`;
- `pose`: float tensor `[T, 25, 2]`, shoulder-centered and shoulder-distance normalized;
- `face`: float tensor `[T, 136]`, 68 facial landmarks flattened and normalized using the corresponding shoulder frame;
- `sample_id`: string.

No file may contain random placeholders. The dataset loader validates dimensions and sequence length and raises on missing files.

## OpenPose input

For each sample, OpenPose JSON files are expected below `<openpose-json-root>/<sample_id>/`. Each frame must contain BODY_25 `pose_keypoints_2d` and at least 68 facial `face_keypoints_2d` points for the selected primary person. The extraction code rejects frames with missing required joints/landmarks rather than filling them with synthetic coordinates.
