# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

umask 007

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ka9vt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ka9vt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ka9vt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ka9vt/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Override the built-in 'cd' command with a custom function.
function cd() 
{
    # Use the built-in 'cd' command to change directories to the one provided as argument.
    # "$@" passes all arguments received by the function to the 'cd' command.
    builtin cd "$@" && 
    { # Execute the following block if 'cd' command succeeds.
        # Check if the current working directory is the project directory for which we want to activate the Conda environment.
        if [[ $PWD == /home/ka9vt/Projects/link-prediction ]]; then
            # If in the correct directory, use Conda to activate the environment.
            # This assumes that the 'conda' command is available and the shell has been initialized for Conda.
            conda activate link_prediction
            # Print a message indicating that the virtual environment has been activated.
            # This is useful for confirming to the user that the environment activation has taken place.
            echo "Activated virtual environment: link_prediction"
        fi # End the condition check.
    } # End of command block to execute after changing directory.
}

