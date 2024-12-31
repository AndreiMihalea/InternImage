num_classes_path = 6 # 2 for ('rest', 'path'); 6 for separate classes based on trajectory angle
dataset_name = 'upb' # upb or kitti
if num_classes_path > 2:
    flip_prob = 0.0
    annotations_loader = 'LoadAnnotationsSplitByCategory'
else:
    flip_prob = 0.5
    annotations_loader = 'LoadAnnotations'