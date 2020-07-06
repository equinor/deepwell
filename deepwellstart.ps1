param (
	[switch]$build = $false,
	[switch]$run = $false,
	[switch]$show = $false,
	[switch]$b = $false,
	[switch]$r = $false,
	[switch]$s = $false,
	[switch]$rs = $false,
	[switch]$br = $false,
	[switch]$brs = $false,
	[switch]$bs = $false
)

if ($build -or $b) {
    docker build -t deepwell-app .
}
elseif ($run -or $r) {
    docker rm -f dwrunning ; docker run -it --mount type=bind,source="$(pwd)",target=/app -p 8080:8080 --name dwrunning deepwell-app
}
elseif ($show -or $s -or $rs) {
    docker rm -f dwrunning ; docker run -dit --mount type=bind,source="$(pwd)",target=/app -p 8080:8080 --name dwrunning deepwell-app ; if ($?){ Start-Sleep -s 2; Start "http://localhost:8080/"; docker logs -f dwrunning }
}
elseif ($br) {
    docker build -t deepwell-app . ; if ($?){docker rm -f dwrunning} ; docker run -it --mount type=bind,source="$(pwd)",target=/app -p 8080:8080 --name dwrunning deepwell-app
}
elseif ($brs -or $bs) {
    docker build -t deepwell-app . ; if ($?){docker rm -f dwrunning} ; docker run -dit --mount type=bind,source="$(pwd)",target=/app -p 8080:8080 --name dwrunning deepwell-app; if ($?){ Start-Sleep -s 2; Start "http://localhost:8080/"; docker logs -f dwrunning }
}
else{
    write-output "No accepted flags detected. Choose from -b,-r,-rs,-br or -brs. b=build, r=run, s=show like: .\deepwellstart.ps1 -br"
}