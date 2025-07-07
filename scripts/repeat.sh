#!/bin/bash

fp8_flag=false
srun_flag=false
pretrained=false

print_help() {
  echo "Usage: bash repeat.sh [options] model_name model_version"
  echo ""
  echo "Arguments:"
  echo "  model_name               Model name (e.g., resnet, vit)."
  echo "  model_version            Model version (e.g., 18, 34, 50)."
  echo ""
  echo "Options:"
  echo "  -h                                   Print this help message and exit."
  echo "  -hh, --deep-help                     Pass '--help' to the Python script for detailed usage."
  echo "  -b NUM, --batch_size NUM             Batch size for training (default: 256)."
  echo "  --pretrained                         Use pretrained model. If provided, is_pretrained is set to true."
  echo "  -d DEVICES, --devices DEVICES        Device to run training on (e.g., 1, '[0]' '[0, 1]')."
  echo "  -ds NAME, --dataset_name NAME        Dataset name (allowed values: rgb, h5; default: rgb)."
  echo "  -dd PATH, --data_dir PATH            Dataset directory path (default: './data/dataset')."
  echo "  --fp8, --no-fp8                      Enable/Disable FP8 precision (default: disabled)"
  echo "  --srun, --no-srun                    Enable/Disable slurm srun (default: disabled)"
  echo "  --lr VALUE                           Learning rate value (required)"
  echo "  -wit NUM, --warmup_iters NUM         Number of warmup iterations (default: 5)."
  echo "  -cit NUM, --cosine_iters NUM         Number of cosine iterations (default: 90)."
  echo "  -kit NUM, --keep_iters NUM           Number of keep iterations (default: 10)."
  echo "  -wfac VALUE, --warmup_factor VALUE   Warmup factor value (default: 0.1)."
  echo "  -emfac VALUE, --eta_min_factor VALUE Eta min factor value (default: 0.01)."
  echo "  -wd VALUE, --weight_decay VALUE      Weight decay value (default: 0.01)."
  echo "  --max_epochs VALUE                   Max epochs value (default: 150)."
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  -h)
    print_help
    ;;
  -hh | --deep-help)
    deep_help="--help"
    shift
    ;;
  -b | --batch_size)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    batch_size="$2"
    shift 2
    ;;
  --pretrained)
    pretrained=true
    shift
    ;;
  -d | --devices)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    devices="$2"
    shift 2
    ;;
  -ds | --dataset_name)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    if [[ "$2" != "rgb" && "$2" != "h5" ]]; then
      echo "Error: Invalid dataset_name '$2'. Allowed values are 'rgb' and 'h5'."
      print_help
      exit 1
    fi
    dataset_name="$2"
    shift 2
    ;;
  -dd | --data_dir)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    data_dir="$2"
    shift 2
    ;;
  --fp8)
    fp8_flag=true
    shift
    ;;
  --no-fp8)
    fp8_flag=false
    shift
    ;;
  --srun)
    srun_flag=true
    shift
    ;;
  --no-srun)
    srun_flag=false
    shift
    ;;
  --lr)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '--lr'."
      print_help
      exit 1
    fi
    lr="$2"
    shift 2
    ;;
  -wit | --warmup_iters)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    warmup_iters="$2"
    shift 2
    ;;
  -cit | --cosine_iters)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    cosine_iters="$2"
    shift 2
    ;;
  -kit | --keep_iters)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    keep_iters="$2"
    shift 2
    ;;
  -wfac | --warmup_factor)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    warmup_factor="$2"
    shift 2
    ;;
  -emfac | --eta_min_factor)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    eta_min_factor="$2"
    shift 2
    ;;
  -wd | --weight_decay)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    weight_decay="$2"
    shift 2
    ;;
  --max_epochs)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    max_epochs="$2"
    shift 2
    ;;
  *)
    if [[ -z "$model_name" ]]; then
      model_name="$1"
    elif [[ -z "$model_version" ]]; then
      model_version="$1"
    else
      echo "Error: Unexpected argument '$1'."
      print_help
      exit 1
    fi
    shift
    ;;
  esac
done

if [[ -z "$model_name" || -z "$model_version" ]]; then
  echo "Error: Missing required parameters (model_name, model_version)."
  print_help
  exit 1
fi

if [[ -z "$dataset_name" ]]; then
  dataset_name="rgb"
fi

if [[ -z "$data_dir" ]]; then
  data_dir="./data/dataset"
  if [[ ! -d "$data_dir" ]]; then
    echo "Error: Data directory '$data_dir' does not exist."
    print_help
    exit 1
  fi
fi

experiment="default_${dataset_name}"

command="python src/repeat.py"
command+=" experiment=${experiment}"
command+=" dataset_name=${dataset_name}"
command+=" paths.data_dir=${data_dir}"
command+=" model=${model_name}"
command+=" model_name=${model_name}"
command+=" model_version=${model_version}"
command+=" warmup_iters=${warmup_iters:-5}"
command+=" cosine_iters=${cosine_iters:-90}"
command+=" keep_iters=${keep_iters:-10}"
command+=" warmup_factor=${warmup_factor:-0.1}"
command+=" eta_min_factor=${eta_min_factor:-0.01}"
command+=" trainer.max_epochs=${max_epochs:-150}"
command+=" model.optimizer.weight_decay=${weight_decay:-0.01}"

if [[ "$pretrained" = true ]]; then
  command+=" is_pretrained=true is_compiled=false"
fi

if [[ -n "$batch_size" ]]; then
  command+=" batch_size=${batch_size}"
fi

if [[ -n "$devices" ]]; then
  if [[ "$devices" =~ ^[0-9]+$ ]]; then
    command+=" trainer.devices=${devices}"
  else
    command+=" trainer.devices='${devices}'"
  fi
fi

if [ "$fp8_flag" = true ]; then
  command+=" trainer.precision='transformer-engine'"
fi

if [ "$srun_flag" = true ]; then
  command="srun $command"
fi

if [[ -z "$lr" ]]; then
  echo "Error: Missing required parameter '--lr'."
  print_help
  exit 1
else
  command+=" lr=${lr}"
fi

if [[ -n "$deep_help" ]]; then
  command+=" $deep_help"
fi

seeds=(0 21 42 84 168 336 672 1344 2688 3407)
# seeds=(21 42 84 168 336 672 1344 2688 3407)

for seed in "${seeds[@]}"; do
  if [[ "$pretrained" = true ]]; then
    output_dir="logs/repeat/pretrained_True/${model_name}/${model_version}/${seed}/checkpoints"
  else
    output_dir="logs/repeat/pretrained_False/${model_name}/${model_version}/${seed}/checkpoints"
  fi

  if [ -d "$output_dir" ]; then
    echo -e "Skipping seed=${seed}: Output directory '$output_dir' already exists."
    # continue
  fi

  # cmd="$command seed=${seed} paths.output_dir=${output_dir}"
  cmd="$command seed=${seed}"
  echo -e "Running with seed=${seed}: \033[32m$cmd\033[0m"
  eval "$cmd"
done
