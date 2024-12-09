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
            echo "        $error_message" >&2
        fi
    done
}

# Temp input file creation
temp_input_file="meep_temp.in"
echo "---------------------"
echo "Creating $temp_input_file:"
echo "---------------------"

> "$temp_input_file"

# Source section
echo ""
echo "Source Section"
source_type=$(ask_with_default "    Enter the source type (gaussian, continuous, chirped, pulse)" "pulse" "^(gaussian|continuous|chirped|pulse)$" "Invalid source type. Please enter one of gaussian, continuous, chirped, or pulse.")

# Handle specific input fields based on source type
case "$source_type" in
    "gaussian")
        wavelength=$(ask_with_default "    Enter the source wavelength" "0.6" "^[0-9]*\.?[0-9]+$" "Invalid wavelength. Please enter a valid number.")
        width=$(ask_with_default "    Enter the source width" "0.05" "^[0-9]*\.?[0-9]+$" "Invalid width. Please enter a valid number.")
        {
            echo "start source"
            echo "    source_type gaussian"
            echo "    wavelength $wavelength"
            echo "    width $width"
            echo "    is_integrated true"
            echo "    sourceCenter -0.04"
            echo "    sourceSize 0 0.1 0.1"
            echo "end source"
        } >> "$temp_input_file"
        ;;
    "continuous")
        wavelength=$(ask_with_default "    Enter the source wavelength" "0.6" "^[0-9]*\.?[0-9]+$" "Invalid wavelength. Please enter a valid number.")
        {
            echo "start source"
            echo "    source_type continuous"
            echo "    wavelength $wavelength"
            echo "    is_integrated true"
            echo "    sourceCenter -0.04"
            echo "    sourceSize 0 0.1 0.1"
            echo "end source"
        } >> "$temp_input_file"
        ;;
    "chirped")
        {
            echo "start source"
            echo "    source_type chirped"
            echo "    sourceCenter -0.04"
            echo "    sourceSize 0 0.1 0.1"
        } >> "$temp_input_file"
        freq_or_wav=$(ask_with_default "    Which parameter do you want to specify? (frequency or wavelength)" "frequency" "^(frequency|wavelength)$" "Please enter 'frequency' or 'wavelength'.")
        if [ "$freq_or_wav" = "frequency" ]; then
            frequency=$(ask_with_default "    Enter the source frequency" "2" "^[0-9]*\.?[0-9]+$" "Invalid frequency. Please enter a valid number.")
            echo "    frequency $frequency" >> "$temp_input_file"
        fi
        if [ "$freq_or_wav" = "wavelength" ]; then
            wavelength=$(ask_with_default "    Enter the source wavelength" "0.5" "^[0-9]*\.?[0-9]+$" "Invalid wavelength. Please enter a valid number.")
            echo "    wavelength $wavelength" >> "$temp_input_file"
        fi
        width=$(ask_with_default "    Enter the source width" "0.2" "^[0-9]*\.?[0-9]+$" "Invalid width. Please enter a valid number.")
        peak_time=$(ask_with_default "    Enter the source peak time" "10" "^[0-9]*\.?[0-9]+$" "Invalid peak time. Please enter a valid number.")
        chirp_rate=$(ask_with_default "    Enter the chirp rate" "-0.5" "^[+-]?[0-9]*\.?[0-9]+$" "Invalid chirp rate. Please enter a valid number.")
        {
            echo "    width $width"
            echo "    peakTime $peak_time"
            echo "    chirpRate $chirp_rate"
            echo "    is_integrated true"
            echo "end source"
        } >> "$temp_input_file"
        ;;
    "pulse")
        {
            echo "start source"
            echo "    source_type pulse"
            echo "    sourceCenter -0.04"
            echo "    sourceSize 0 0.1 0.1"
        } >> "$temp_input_file"
        freq_or_wav=$(ask_with_default "    Which parameter do you want to specify? (frequency or wavelength)" "frequency" "^(frequency|wavelength)$" "Please enter 'frequency' or 'wavelength'.")
        if [ "$freq_or_wav" = "frequency" ]; then
            frequency=$(ask_with_default "    Enter the source frequency" "2" "^[0-9]*\.?[0-9]+$" "Invalid frequency. Please enter a valid number.")
            echo "    frequency $frequency" >> "$temp_input_file"
        fi
        if [ "$freq_or_wav" = "wavelength" ]; then
            wavelength=$(ask_with_default "    Enter the source wavelength" "0.5" "^[0-9]*\.?[0-9]+$" "Invalid wavelength. Please enter a valid number.")
            echo "    wavelength $wavelength" >> "$temp_input_file"
        fi
        width=$(ask_with_default "    Enter the source width" "0.2" "^[0-9]*\.?[0-9]+$" "Invalid width. Please enter a valid number.")
        peak_time=$(ask_with_default "    Enter the source peak time" "10" "^[0-9]*\.?[0-9]+$" "Invalid peak time. Please enter a valid number.")
        {
            echo "    width $width"
            echo "    peakTime $peak_time"
            echo "    is_integrated true"
            echo "end source"
        } >> "$temp_input_file"
        ;;
esac
# First part of Simulation Section
echo ""
echo "Simulation Section"
resolution=$(ask_with_default "    Enter the simulation resolution" "1000" "^[1-9][0-9]{3,}$" "Resolution must be a number greater than 1000.")
total_time=$(ask_with_default "    Enter the total simulation time" "50" "^[0-9]+$" "Invalid time format. Must be a number.")
total_time_unit=$(ask_with_default "    Enter the total simulation time unit (fs, as, mu, au)" "fs" "^(fs|as|mu|au)$" "Invalid time unit. Must be one of fs, as, mu, or au.")

# Simulation section (always included)
{
    echo ""
    echo "start simulation"
    echo "    resolution $resolution"
    echo "    responseCutOff 1e-12"
    echo "    cellLength 0.1"
    echo "    pmlThickness 0.01"
    echo "    totalTime $total_time $total_time_unit"
    echo "    symmetries Y 1 Z -1"
    echo "    surroundingMaterialIndex 1.33"
    echo "    directionCalculation z"
    echo "end simulation"
} >>"$temp_input_file"

# Optionally include molecule section
echo ""
echo "Molecule Section"
include_molecule=$(ask_with_default "    Do you want to include a molecule section? (y/n)" "n" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$include_molecule" = "y" ]; then
    {
        echo ""
        echo "start molecule"
        echo "    center 0 0 0"
        echo "    directionCalculation z"
        echo "end molecule"
    } >>"$temp_input_file"
fi

# outputPNG section
echo ""
echo "outputPNG Section"
include_outputPNG=$(ask_with_default "    Do you want to include an outputPNG section? (y/n)" "n" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$include_outputPNG" = "y" ]; then
    imageDirName=$(ask_with_default "    Enter the name of the directory to output the images" "imgDir" "^[a-zA-Z0-9]+$" "Please enter a valid name.")
    timestepsBetween=$(ask_with_default "    Enter the number of timesteps between each screenshot" "5" "^[0-9]+$" "Please enter a valid number.")
    {
        echo "start outputPNG" 
        echo "    imageDirName $imageDirName"
        echo "    timestepsBetween $timestepsBetween"
        echo "    intensityMin -3"
        echo "    intensityMax 3"
        echo "end outputPNG"
    } >>"$temp_input_file"
fi

case "$source_type" in
    "gaussian")
        prefix="molecule-Files/GaussianTests/"
        dir_name="gauss_w${width}_l${wavelength}_r${resolution}_tT${total_time}"
        ;;
    "continuous")
        prefix="molecule-Files/ContinuousTests/"
        dir_name="cont_w${width}_l${wavelength}_r${resolution}_tT${total_time}"
        ;;
    "chirped")
        prefix="molecule-Files/ChirpedTests/"
        chirp_rate_prefix=$(echo "$chirp_rate" | awk '{if ($1 < 0) print "n"; else print ""}')
        chirp_rate_abs=$(echo "$chirp_rate" | sed 's/^-//')
        dir_name="chirped_f${frequency}_w${width}_pT${peak_time}_cR${chirp_rate_prefix}${chirp_rate_abs}_r${resolution}_tT${total_time}"
        ;;
    "pulse")
        prefix="molecule-Files/PulseTests/"
        if [ "$freq_or_wav" = "wavelength" ]; then
            dir_name="pulse_w${wavelength}_w${width}_pT${peak_time}_r${resolution}_tT${total_time}"
        fi
        if [ "$freq_or_wav" = "frequency" ]; then
                dir_name="pulse_f${frequency}_w${width}_pT${peak_time}_r${resolution}_tT${total_time}"
        fi
        ;;
esac

# Optionally include matplotlib section
echo ""
echo "Matplotlib Section"
include_matplotlib=$(ask_with_default "    Do you want to include a matplotlib section? (y/n)" "y" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$include_matplotlib" = "y" ]; then
    output_name=$(ask_with_default "    Enter the output file name" "$dir_name" "^[a-zA-Z0-9_.]+$" "Please enter a valid name.")
    {
        echo ""
        echo "start matplotlib"
        echo "    output $output_name"
        echo "end matplotlib"
    } >>"$temp_input_file"
fi

# Directory creation just before SLURM submission
echo ""
echo "-------------------------------"
echo "Creating directory:"
echo "    $prefix$dir_name"
echo "-------------------------------"

mkdir -p "$prefix$dir_name"
cp molecule-Files/files/* "$prefix$dir_name"
cd "$prefix$dir_name" || exit

# Move temp input file to the new directory
mv "../../../$temp_input_file" "meep.in"

# SLURM submit script creation
submit_file="submit_plasmol.sh"
echo ""
echo "------------------------"
echo "Generating $submit_file:"
echo "------------------------"

echo ""
echo "Submission Script Section"
memory=$(ask_with_default "    Enter memory allocation (e.g., 50G)" "50G" "^[0-9]+[A-Za-z]+$" "Please enter a valid memory format (e.g., 50G).")
time_limit=$(ask_with_default "    Enter the time limit (e.g., 14-00:00:00)" "14-00:00:00" "^[0-9]{2}-[0-9]{2}:[0-9]{2}:[0-9]{2}$" "Please enter a valid time limit format (e.g., 14-00:00:00).")

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

# SLURM job submission
echo ""
submit_choice=$(ask_with_default "    Do you want to submit the job to SLURM now? (y/n)" "y" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$submit_choice" = "y" ]; then
    sbatch "$submit_file"
    echo "Job submitted."
else
    echo "Job not submitted. You can submit manually using: sbatch $submit_file"
fi
