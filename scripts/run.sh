#!/bin/bash

fp8_flag=false

print_help() {
  echo "Usage: bash repeat.sh [options] model_name model_version"
  echo ""
  echo "Arguments:"
  echo "  model_name               Model name (e.g., resnet, vit)."
  echo "  model_version            Model version (e.g., 18, 34, 50)."
  echo ""
  echo "Options:"
  echo "  -b NUM, --batch_size NUM          Batch size for training (default: 256)."
  echo "  -p PATH, --pretrained_model_path  PATH Path to the pretrained model. If provided, is_pretrained is set to true."
  echo "  -d DEVICES, --devices DEVICES     Device to run training on (e.g., 1, '[0]' '[0, 1]')."
  echo "  -ds NAME, --dataset_name NAME     Dataset name (allowed values: rgb, h5; default: rgb)."
  echo "  -dd PATH, --data_dir PATH         Dataset directory path (default: './data/dataset')."
  echo "  --fp8, --no-fp8                   Enable/Disable FP8 precision (default: enabled)"
  echo "  --lr VALUE                        Learning rate value (required)"
  echo "  -h                                Print this help message and exit."
  echo "  -hh, --deep-help                  Pass '--help' to the Python script for detailed usage."
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
  -p | --pretrained_model_path)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '$1'."
      print_help
      exit 1
    fi
    pretrained_model_path="$2"
    shift 2
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
  --lr)
    if [[ -z "$2" || "$2" == -* ]]; then
      echo "Error: Missing value for option '--lr'."
      print_help
      exit 1
    fi
    lr="$2"
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

command="python src/train.py"
command+=" experiment=${experiment}"
command+=" dataset_name=${dataset_name}"
command+=" paths.data_dir=${data_dir}"
command+=" model=${model_name}"
command+=" model_name=${model_name}"
command+=" model_version=${model_version}"

if [[ -n "$pretrained_model_path" ]]; then
  command+=" is_pretrained=true"
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

if [[ -n "$deep_help" ]]; then
  command+=" $deep_help"
fi

if [ "$fp8_flag" = true ]; then
  command+=" trainer.precision='transformer-engine'"
fi

if [[ -z "$lr" ]]; then
  echo "Error: Missing required parameter '--lr'."
  print_help
  exit 1
else
  command+=" lr=${lr}"
fi

seed=42

cmd="$command seed=${seed}"
echo -e "Running with seed=${seed}: \033[32m$cmd\033[0m"
eval "$cmd"
