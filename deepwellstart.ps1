param (
	[switch]$build = $false,
	[switch]$run = $false,
	[switch]$b = $false,
	[switch]$r = $false,
	[switch]$br = $false,
	$Textparamforpython=$args[0]
)

if($Textparamforpython-eq $null) {		#This is to allow ".\deepwellstart.ps1 -r" to be run without any parameters and not cause problems in python
  $Textparamforpython = " "
}


function build-container {
	docker build -t deepwell-app . ; if ($?) { write-output "Deepwell container built successfully" }else { write-output "Something wrong when building deepwell container" }
}

function start-container {
	docker rm -f dwrunning
	docker run -dit --mount type=bind,source="$(pwd)",target=/app -p 0.0.0.0:7007:6006 -p 8080:8080 --name dwrunning deepwell-app ; if ($?) { write-output "Initialized deepwell container successfully" } else { write-output "Something went wrong when trying to run (initialize) the container" }
}

function start-tensorboard-server {
	docker exec -dit dwrunning tensorboard --logdir /usr/src/app/logs/ --host 0.0.0.0 --port 6006; if ($?) { "Tensorboard server started successfully. Running agent..." } else { "Something went wrong when trying to start tensorboard server" }
}

function start-python-code {
	Param($Textparamforpython)
	docker exec -it dwrunning python /app/main.py $Textparamforpython
}

function start-whole-container {
	Param([string]$Textparamforpython)
	start-container ; if ($?) { start-tensorboard-server } ; if ($?) { start-python-code($Textparamforpython,$Textparamforpythontwo) }
}

if ( $build -or $b ) { build-container } 
elseif ( $br ) { build-container ; start-whole-container($Textparamforpython) }
elseif ($run -or $r) { start-whole-container($Textparamforpython) }
else { write-output "No accepted flags detected. Choose from -b,-r,-rs,-br or -brs. b=build, r=run, s=show like: .\deepwellstart.ps1 -br" }

#The string Textparamforpython is used in in run_dw_env.py to determine what block of code should run. Examples could be train, load or retrain



