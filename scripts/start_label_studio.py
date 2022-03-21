import yaml
from subprocess import run, Popen

def start_labeling_job(trainval_dir):
    cmd = ["chmod", "+x", "./labels/get_urls.sh"]
    run(cmd)

    cmd = ["./labels/get_urls.sh", f"{trainval_dir}", "*.jpg"]
    Popen(cmd)

    cmd = ["label-studio", "start"]
    Popen(cmd)

if __name__ == "__main__":
    with open('./params.yaml', 'r') as params_file:
        params = yaml.safe_load(params_file)
        points_path = params['data']['points_path']

    trainval_dir = Path(params['slices']['trainval_dir'])/project_name/'sliced_images'
    start_labeling_job(trainval_dir)