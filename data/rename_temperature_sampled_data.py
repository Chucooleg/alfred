import os
import progressbar


for path, directories, files in os.walk('/root/data_alfred/json_feat_2.1.0_backup_20200826_agent_training'):

    for file in progressbar.progressbar(files):
        if ('aug_baseline' in file or 'aug_explainer' in file) and '.json' in file:
            src_file_path = os.path.join(path, file)
            dest_file_path = os.path.join(path, '_'.join(file.split('_')[:3]) + '.json')

            assert os.path.exists(src_file_path)
            os.rename(src_file_path, dest_file_path)
            assert os.path.exists(dest_file_path)
            # print (os.path.join(path, files))