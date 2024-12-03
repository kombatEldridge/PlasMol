#!/bin/bash

# Ask questions with defaults and validate inputs
ask_with_default() {
    local prompt="$1"
    local default="$2"
    local regex="$3"
    local error_message="$4"
    
    while true; do
        read -p "$prompt [$default]: " input
        input="${input:-$default}"
        
        if [[ "$input" =~ $regex ]]; then
            echo "$input"
            break
        else
            echo "$error_message"
        fi
    done
}

# Input file creation
input_file="meep.in"
echo "Creating $input_file..."

# Source section
echo ""
echo "Source Section"
frequency=$(ask_with_default "Enter the source frequency" "1" "^[0-9]*\.?[0-9]+$" "Invalid frequency. Please enter a valid number.")
width=$(ask_with_default "Enter the source width" "0.1" "^[0-9]*\.?[0-9]+$" "Invalid width. Please enter a valid number.")
peak_time=$(ask_with_default "Enter the source peak time" "5" "^[0-9]*\.?[0-9]+$" "Invalid peak time. Please enter a valid number.")
chirp_rate=$(ask_with_default "Enter the chirp rate" "-0.5" "^[+-]?[0-9]*\.?[0-9]+$" "Invalid chirp rate. Please enter a valid number.")

# First part of Simulation Section
echo ""
echo "Simulation Section"
resolution=$(ask_with_default "Enter the simulation resolution" "8000" "^[1-9][0-9]{3,}$" "Resolution must be a number greater than 1000.")
total_time=$(ask_with_default "Enter the total simulation time (e.g., 40 fs)" "40 fs" "^[0-9]+( [a-zA-Z]+)?$" "Invalid time format. Must be a number followed by a time unit (fs, as, mu, or au).")

# Generate directory name based on source parameters
chirp_rate_prefix=$(echo "$chirp_rate" | awk '{if ($1 < 0) print "n"; else print ""}')
chirp_rate_abs=$(echo "$chirp_rate" | sed 's/^-//')
time_value=$(echo "$total_time" | sed -E 's/[^0-9]*([0-9]+).*/\1/')
dir_name="/project/bldrdge1/PlasMol/molecule-Files/chirpedPulse-Test/f${frequency}_w${width}_pT${peak_time}_cR${chirp_rate_prefix}${chirp_rate_abs}_r${resolution}_tT${time_value}"

mkdir -p "$dir_name"
cp /project/bldrdge1/PlasMol/molecule-Files/files/* "$dir_name"
cd "$dir_name" || exit

# Input file creation
{
    echo "start source"
    echo "    source_type chirped"
    echo "    sourceCenter -0.04"
    echo "    sourceSize 0 0.1 0.1"
    echo "    frequency $frequency"
    echo "    width $width"
    echo "    peakTime $peak_time"
    echo "    chirpRate $chirp_rate"
    echo "    is_integrated True"
    echo "end source"
} >"$input_file"

# Simulation section (always included)
{
    echo ""
    echo "start simulation"
    echo "    resolution $resolution"
    echo "    responseCutOff 1e-12"
    echo "    cellLength 0.1"
    echo "    pmlThickness 0.01"
    echo "    totalTime $total_time"
    echo "    symmetries Y 1 Z -1"
    echo "    surroundingMaterialIndex 1.33"
    echo "    directionCalculation z"
    echo "end simulation"
} >>"$input_file"


# Optionally include molecule section
echo ""
echo "Molecule Section"
include_molecule=$(ask_with_default "Do you want to include a molecule section? (y/n)" "y" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$include_molecule" = "y" ]; then
    {
        echo ""
        echo "start molecule"
        echo "    center 0 0 0"
        echo "    directionCalculation z"
        echo "end molecule"
    } >>"$input_file"
fi

# outputPNG section
echo ""
echo "outputPNG Section"
include_outputPNG=$(ask_with_default "Do you want to include an outputPNG section? (y/n)" "n" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$include_outputPNG" = "y" ]; then
    imageDirName=$(ask_with_default "Enter the name of the directory to output the images" "imgDir")
    timestepsBetween=$(ask_with_default "Enter the number of timesteps between each screenshot" "5" "^[0-9]+$" "Please enter a valid number.")
    {
        echo "start outputPNG" 
        echo "    imageDirName $imageDirName"
        echo "    timestepsBetween $timestepsBetween"
        echo "    intensityMin -3"
        echo "    intensityMax 3"
        echo "end outputPNG"
    } >>"$input_file"
fi

# Optionally include matplotlib section
echo ""
echo "Matplotlib Section"
include_matplotlib=$(ask_with_default "Do you want to include a matplotlib section? (y/n)" "y" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$include_matplotlib" = "y" ]; then
    time_value=$(echo "$total_time" | sed 's/ //g')
    default_output_name="chirped${time_value}"
    output_name=$(ask_with_default "Enter the output file name" "$default_output_name")
    {
        echo ""
        echo "start matplotlib"
        echo "    output $output_name"
        echo "end matplotlib"
    } >>"$input_file"
fi

# SLURM submit script creation
submit_file="submit_plasmol.sh"
echo ""
echo "Generating $submit_file..."

echo ""
echo "Submission Script Section"
memory=$(ask_with_default "Enter memory allocation (e.g., 50G)" "50G" "^[0-9]+[A-Za-z]+$" "Please enter a valid memory format (e.g., 50G).")
time_limit=$(ask_with_default "Enter the time limit (e.g., 14-00:00:00)" "14-00:00:00" "^[0-9]{2}-[0-9]{2}:[0-9]{2}:[0-9]{2}$" "Please enter a valid time limit format (e.g., 14-00:00:00).")

# Write to submit file
cat <<EOL >"$submit_file"
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --partition=acomputeq
#SBATCH --job-name=plasmol
#SBATCH --time=$time_limit
#SBATCH --export=NONE

export QT_QPA_PLATFORM="minimal"

module load meep/1.29
/opt/ohpc/pub/apps/uofm/python/3.9.13/bin/python3 /project/bldrdge1/PlasMol/bohr_dev/driver.py -m meep.in -b pyridine.in -vv -l plasmol_hpc.log

EOL

echo ""
echo "Directory created:\n\t$dir_name"

# SLURM job submission
submit_choice=$(ask_with_default "Do you want to submit the job to SLURM now? (y/n)" "y" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$submit_choice" = "y" ]; then
    sbatch "$submit_file"
    echo "Job submitted."
else
    echo "Job not submitted. You can submit manually using: sbatch $submit_file"
fi