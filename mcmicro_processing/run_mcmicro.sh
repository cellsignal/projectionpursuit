#!/bin/sh
#SBATCH --job-name="mcmicro_signalstar"
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=8gb

module load devel/java_jdk/1.18
module load system/singularity/3.11.3

project_dir=/gpfs/bwfor/work/ws/hd_hl269-signalstar/

nextflow run schapirolabor/mcmicro -r 'staging' \
    --in $project_dir/data \
    --params $project_dir/params.yml \
    -profile singularity \
    -c $project_dir/run_mcmicro.config \
    -with-report $project_dir/execution_report.html