param (
	[switch]$build = $false,
	[switch]$run = $false,
	[switch]$b = $false,
	[switch]$r = $false,
	[switch]$br = $false,
	$Textparamforpython=$args[0],
	$Numparamforpython=$args[1],
	$Filenameparamforpython=$args[2],
	$Agentnameparamforpython=$args[3]
)

if($Textparamforpython-eq $null) {		#This is to allow ".\deepwellstart.ps1 -r" to be run without any parameters and not cause problems in python
  $Textparamforpython = " "
}

if($Numparamforpython-eq $null) {		#This is to allow ".\deepwellstart.ps1 -r" to be run without any parameters and not cause problems in python
  $Numparamforpython = 10000			#Default number of timesteps if none is specified
}

if($Filenameparamforpython-eq $null) {		#This is to allow ".\deepwellstart.ps1 -r" to be run without any parameters and not cause problems in python
  $Filenameparamforpython = " "		#Default number of timesteps if none is specified
}

if($Agentnameparamforpython-eq $null) {		#This is to allow ".\deepwellstart.ps1 -r" to be run without any parameters and not cause problems in python
	$Agentnameparamforpython = " "	#If agent name not specified
}


function build-container {
	docker build -t deepwell-app . ; if ($?) { write-output "Deepwell container built successfully" }else { write-output "Something wrong when building deepwell container" }
}

function start-container {
	docker rm -f dwrunning
	docker run -dit --mount type=bind,source="$(pwd)",target=/usr/src/app -p 0.0.0.0:7007:6006 -p 0.0.0.0:8080:8080 --name dwrunning deepwell-app ; if ($?) { write-output "Initialized deepwell container successfully" } else { write-output "Something went wrong when trying to run (initialize) the container" }
}

function start-tensorboard-server {
	docker exec -dit dwrunning tensorboard --logdir /usr/src/app/tensorboard_logs/ --host 0.0.0.0 --port 6006; if ($?) { "Tensorboard server started successfully. Running agent..." } else { "Something went wrong when trying to start tensorboard server" }
}

function start-python-code($Textparamforpython, $Numparamforpython, $Filenameparamforpython, $Agentnameparamforpython ) {
	docker exec -it dwrunning python /usr/src/app/main.py $Textparamforpython $Numparamforpython $Filenameparamforpython $Agentnameparamforpython
}

function start-whole-container($Textparamforpython, $Numparamforpython, $Filenameparamforpython, $Agentnameparamforpython) {
	start-container ; if ($?) { start-tensorboard-server } ; if ($?) { start-python-code $Textparamforpython $Numparamforpython $Filenameparamforpython $Agentnameparamforpython }
}

if ( $build -or $b ) { build-container } 
elseif ( $br ) { build-container ; start-whole-container $Textparamforpython $Numparamforpython $Filenameparamforpython $Agentnameparamforpython }
elseif ($run -or $r) { start-whole-container $Textparamforpython $Numparamforpython $Filenameparamforpython $Agentnameparamforpython }
else { write-output "No accepted flags detected. Choose from -build,-run,-b,-r or -br like: '.\deepwellstart.ps1 -br'. To specify behavior in run_dw_env.py run with parameters like: '.\deepwellstart.ps1 -r train 10000'" }

#The string Textparamforpython is used in in run_dw_env.py to determine what block of code should run. Examples could be train, load or retrain



