#!/bin/bash

cd ../

# Defaults file
DEFAULTS_FILE="helpers/defaults.conf"
LAST_RUN_FILE="helpers/last_run.conf"

# Load defaults from file
load_defaults() {
    if [[ -f "$DEFAULTS_FILE" ]]; then
        source "$DEFAULTS_FILE"
    else
        echo "No defaults file found."
    fi
}

# Function to save current defaults to the file
save_defaults() {
    echo "Saving current settings to $LAST_RUN_FILE..."
    cat > "$LAST_RUN_FILE" <<EOL
source_type=$source_type

wavelength_gaussian=$wavelength_gaussian
width_gaussian=$width_gaussian

wavelength_continuous=$wavelength_continuous

freq_or_wav=$freq_or_wav

frequency_chirped=$frequency_chirped
wavelength_chirped=$wavelength_chirped
width_chirped=$width_chirped
peakTime_chirped=$peakTime_chirped
chirpRate_chirped=$chirpRate_chirped

frequency_pulse=$frequency_pulse
wavelength_pulse=$wavelength_pulse
width_pulse=$width_pulse
peakTime_pulse=$peakTime_pulse

resolution=$resolution
total_time=$total_time
total_time_unit=$total_time_unit

include_molecule=$include_molecule
molecule=$molecule

include_outputPNG=$include_outputPNG
imageDirName=$imageDirName
timestepsBetween=$timestepsBetween

include_matplotlib=$include_matplotlib

submit_choice=$submit_choice
memory=$memory
time_limit=$time_limit
EOL
}

# Load defaults
load_defaults "$DEFAULTS_FILE"

# Ask user if they want to use last run settings as defaults
if [ -f "$LAST_RUN_FILE" ]; then
    read -p "Use last run settings as defaults? (y/n) [n]: " use_last
    use_last="${use_last:-n}"
    if [ "$use_last" == "y" ]; then
        source "$LAST_RUN_FILE"
    fi
fi


# Function to ask questions with defaults
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
source_type=$(ask_with_default "    Enter the source type (gaussian, continuous, chirped, pulse)" "$source_type" "^(gaussian|continuous|chirped|pulse)$" "Invalid source type. Please enter one of gaussian, continuous, chirped, or pulse.")

# Handle specific input fields based on source type
case "$source_type" in
    "gaussian")
        wavelength_gaussian=$(ask_with_default "    Enter the source wavelength" "$wavelength_gaussian" "^[0-9]*\.?[0-9]+$" "Invalid wavelength. Please enter a valid number.")
        width_gaussian=$(ask_with_default "    Enter the source width" "$width_gaussian" "^[0-9]*\.?[0-9]+$" "Invalid width. Please enter a valid number.")
        {
            echo "start source"
            echo "    source_type gaussian"
            echo "    wavelength $wavelength_gaussian"
            echo "    width $width_gaussian"
            echo "    is_integrated true"
            echo "    sourceCenter -0.04"
            echo "    sourceSize 0 0.1 0.1"
            echo "end source"
        } >> "$temp_input_file"
        ;;
    "continuous")
        wavelength_continuous=$(ask_with_default "    Enter the source wavelength" "$wavelength_continuous" "^[0-9]*\.?[0-9]+$" "Invalid wavelength. Please enter a valid number.")
        {
            echo "start source"
            echo "    source_type continuous"
            echo "    wavelength $wavelength_continuous"
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
        freq_or_wav=$(ask_with_default "    Which parameter do you want to specify? (frequency or wavelength)" "$freq_or_wav" "^(frequency|wavelength)$" "Please enter 'frequency' or 'wavelength'.")
        if [ "$freq_or_wav" = "frequency" ]; then
            frequency_chirped=$(ask_with_default "    Enter the source frequency" "$frequency_chirped" "^[0-9]*\.?[0-9]+$" "Invalid frequency. Please enter a valid number.")
            echo "    frequency $frequency_chirped" >> "$temp_input_file"
        fi
        if [ "$freq_or_wav" = "wavelength" ]; then
            wavelength_chirped=$(ask_with_default "    Enter the source wavelength" "$wavelength_chirped" "^[0-9]*\.?[0-9]+$" "Invalid wavelength. Please enter a valid number.")
            echo "    wavelength $wavelength_chirped" >> "$temp_input_file"
        fi
        width_chirped=$(ask_with_default "    Enter the source width" "$width_chirped" "^[0-9]*\.?[0-9]+$" "Invalid width. Please enter a valid number.")
        peakTime_chirped=$(ask_with_default "    Enter the source peak time" "$peakTime_chirped" "^[0-9]*\.?[0-9]+$" "Invalid peak time. Please enter a valid number.")
        chirpRate_chirped=$(ask_with_default "    Enter the chirp rate" "$chirpRate_chirped" "^[+-]?[0-9]*\.?[0-9]+$" "Invalid chirp rate. Please enter a valid number.")
        {
            echo "    width $width_chirped"
            echo "    peakTime $peakTime_chirped"
            echo "    chirpRate $chirpRate_chirped"
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
        freq_or_wav=$(ask_with_default "    Which parameter do you want to specify? (frequency or wavelength)" "$freq_or_wav" "^(frequency|wavelength)$" "Please enter 'frequency' or 'wavelength'.")
        if [ "$freq_or_wav" = "frequency" ]; then
            frequency_pulse=$(ask_with_default "    Enter the source frequency" "$frequency_pulse" "^[0-9]*\.?[0-9]+$" "Invalid frequency. Please enter a valid number.")
            echo "    frequency $frequency_pulse" >> "$temp_input_file"
        fi
        if [ "$freq_or_wav" = "wavelength" ]; then
            wavelength_pulse=$(ask_with_default "    Enter the source wavelength" "$wavelength_pulse" "^[0-9]*\.?[0-9]+$" "Invalid wavelength. Please enter a valid number.")
            echo "    wavelength $wavelength_pulse" >> "$temp_input_file"
        fi
        width_pulse=$(ask_with_default "    Enter the source width" "$width_pulse" "^[0-9]*\.?[0-9]+$" "Invalid width. Please enter a valid number.")
        peakTime_pulse=$(ask_with_default "    Enter the source peak time" "$peakTime_pulse" "^[0-9]*\.?[0-9]+$" "Invalid peak time. Please enter a valid number.")
        {
            echo "    width $width_pulse"
            echo "    peakTime $peakTime_pulse"
            echo "    is_integrated true"
            echo "end source"
        } >> "$temp_input_file"
        ;;
esac
# First part of Simulation Section
echo ""
echo "Simulation Section"
resolution=$(ask_with_default "    Enter the simulation resolution" "$resolution" "^[1-9][0-9]{3,}$" "Resolution must be a number greater than 1000.")
total_time=$(ask_with_default "    Enter the total simulation time" "$total_time" "^[0-9]+$" "Invalid time format. Must be a number.")
total_time_unit=$(ask_with_default "    Enter the total simulation time unit (fs, as, mu, au)" "$total_time_unit" "^(fs|as|mu|au)$" "Invalid time unit. Must be one of fs, as, mu, or au.")

# Simulation section (always included)
{
    echo ""
    echo "start simulation"
    echo "    resolution $resolution"
    echo "    responseCutOff 1e-20"
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
include_molecule=$(ask_with_default "    Do you want to include a molecule section? (y/n)" "$include_molecule" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$include_molecule" = "y" ]; then
    molecule=$(ask_with_default "    What is the name of your molecule input file?" "$molecule" "^[a-zA-Z0-9.]+$" "Answer must be exact name of input file found in moleculeFiles/.")
    if [ ! -e "moleculeFiles/$molecule" ]; then 
        echo "        The file '$molecule' does not exist or is not in moleculeFiles"
        exit 1
    fi
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
include_outputPNG=$(ask_with_default "    Do you want to include an outputPNG section? (y/n)" "$include_outputPNG" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$include_outputPNG" = "y" ]; then
    imageDirName=$(ask_with_default "    Enter the name of the directory to output the images" "$imageDirName" "^[a-zA-Z0-9]+$" "Please enter a valid name.")
    timestepsBetween=$(ask_with_default "    Enter the number of timesteps between each screenshot" "$timestepsBetween" "^[0-9]+$" "Please enter a valid number.")
    {
        echo "start outputPNG" 
        echo "    imageDirName $imageDirName"
        echo "    timestepsBetween $timestepsBetween"
        echo "    intensityMin -3"
        echo "    intensityMax 3"
        echo "end outputPNG"
    } >>"$temp_input_file"
fi

if [ -n "$molecule" ]; then
    left_part=$(echo "$molecule" | sed -E 's/\..*//')
else
    left_part="no-molecule"
fi

case "$source_type" in
    "gaussian")
        prefix="jobs/$left_part/"
        dir_name="gauss_w${width_gaussian}_l${wavelength_gaussian}_r${resolution}_tT${total_time}"
        ;;
    "continuous")
        prefix="jobs/$left_part/"
        dir_name="cont_w${wavelength_continuous}_r${resolution}_tT${total_time}"
        ;;
    "chirped")
        prefix="jobs/$left_part/"
        chirp_rate_prefix=$(echo "$chirpRate_chirped" | awk '{if ($1 < 0) print "n"; else print ""}')
        chirp_rate_abs=$(echo "$chirpRate_chirped" | sed 's/^-//')
        dir_name="chirped_f${frequency_chirped}_w${width_chirped}_pT${peakTime_chirped}_cR${chirp_rate_prefix}${chirp_rate_abs}_r${resolution}_tT${total_time}"
        ;;
    "pulse")
        prefix="jobs/$left_part/"
        if [ "$freq_or_wav" = "wavelength" ]; then
            dir_name="pulse_w${wavelength_pulse}_w${width_pulse}_pT${peakTime_pulse}_r${resolution}_tT${total_time}"
        fi
        if [ "$freq_or_wav" = "frequency" ]; then
                dir_name="pulse_f${frequency_pulse}_w${width_pulse}_pT${peakTime_pulse}_r${resolution}_tT${total_time}"
        fi
        ;;
esac

# Optionally include matplotlib section
echo ""
echo "Matplotlib Section"
include_matplotlib=$(ask_with_default "    Do you want to include a matplotlib section? (y/n)" "$include_matplotlib" "^(y|n)$" "Please enter 'y' or 'n'.")
if [ "$include_matplotlib" = "y" ]; then
    output_name=$(ask_with_default "    Enter the output file name" "$dir_name" "^[a-zA-Z0-9_.]+$" "Please enter a valid name.")
    {
        echo ""
        echo "start matplotlib"
        echo "    output $output_name"
        echo "end matplotlib"
    } >>"$temp_input_file"
fi

save_defaults

# Directory creation just before SLURM submission
echo ""
echo "-------------------------------"
echo "Creating directory:"
echo "    $prefix$dir_name"
echo "-------------------------------"

mkdir -p "$prefix$dir_name"
cp moleculeFiles/* "$prefix$dir_name"
cd "$prefix$dir_name" || exit

# Move temp input file to the new directory
mv "../../../$temp_input_file" "meep.in"

submit_choice=$(ask_with_default "    How would you like to run this job? (slurm/cli/n)" "$submit_choice" "^(slurm|cli|n)$" "Please enter 'slurm', 'cli', or 'n'.")
if [ "$submit_choice" = "slurm" ]; then
    # SLURM submit script creation
    submit_file="submit_plasmol.sh"
    echo ""
    echo "------------------------"
    echo "Generating $submit_file:"
    echo "------------------------"

    echo ""
    echo "Submission Script Section"
    memory=$(ask_with_default "    Enter memory allocation (e.g., 50G)" "$memory" "^[0-9]+[A-Za-z]+$" "Please enter a valid memory format (e.g., 50G).")
    time_limit=$(ask_with_default "    Enter the time limit (e.g., 14-00:00:00)" "$time_limit" "^(\d{1,2}-)?\d{2}:\d{2}:\d{2}$" "Please enter a valid time limit format (e.g., 14-00:00:00).")

    # Write to submit file
    {
        echo "#!/bin/bash"
        echo "#SBATCH --mem=$memory"
        echo "#SBATCH --partition=acomputeq"
        echo "#SBATCH --job-name=plasmol"
        echo "#SBATCH --time=$time_limit"
        echo "#SBATCH --export=NONE"
        echo ""
        echo "export QT_QPA_PLATFORM="minimal""
        echo ""
        echo "module load meep/1.29"
        echo "/opt/ohpc/pub/apps/uofm/python/3.9.13/bin/python3 /project/bldrdge1/PlasMol/bohr/driver.py -m meep.in -b $molecule -vv -l plasmol_hpc.log"
    } >>"$submit_file"
    
    sbatch "$submit_file"
    echo "Job submitted."
elif [ "$submit_choice" = "cli" ]; then
    path=$(pwd)
    echo "Run this command: "
    echo "cd $path && meep && meepy /Users/bldrdge1/Downloads/repos/PlasMol/bohr/driver.py -m meep.in -b pyridine.in -vv -l plasmol_hpc.log"
else
    echo "Job not submitted."
fi
