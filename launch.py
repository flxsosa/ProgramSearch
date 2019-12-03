from datetime import datetime
import time
import subprocess
import os
import json
import sys

ZONE = None

def user():
    import getpass
    return getpass.getuser()


def branch():
    return subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode("utf-8").strip()

def launchGoogleCloud(size, name, gpuType, gpuCount):

    name = name.replace('_','-').replace('.','-').lower()
    snapshot = "max-abstraction-repl"

    #os.system(f"gcloud compute --project tenenbaumlab disks create {name} --size 30 --zone us-east1-b --source-snapshot dreamcoder-jan26 --type pd-standard")
    os.system(f"gcloud compute --project tenenbaumlab disks create {name} --size 100 --zone us-east1-c --source-snapshot {snapshot} --type pd-standard")

    # output = \
    #     subprocess.check_output(["/bin/bash", "-c",
    #                          f"gcloud compute --project=tenenbaumlab instances create {name} --zone=us-east1-b --machine-type={size} --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=150557817012-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --disk=name={name},device-name={name},mode=rw,boot=yes,auto-delete=yes"])
    #POLICY??
    if gpuType:
        if not gpuCount: gpuCount = 1 #default to 1
        accelerator = f"--accelerator=type=nvidia-tesla-{gpuType},count={gpuCount}"
    else:
        accelerator = ""
    output = \
        subprocess.check_output(["/bin/bash", "-c",
                             f"gcloud compute --project=tenenbaumlab instances create {name} --zone=us-east1-c --machine-type={size} --subnet=default --network-tier=PREMIUM --maintenance-policy=TERMINATE --service-account=150557817012-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append {accelerator} --disk=name={name},device-name={name},mode=rw,boot=yes,auto-delete=yes"])
    
    global ZONE
    ZONE = output.decode("utf-8").split("\n")[1].split()[1]
    print(f"Launching in zone {ZONE}")
    return name, name

def scp(address, localFile, remoteFile):
    global ZONE
    command = f"gcloud compute scp --zone={ZONE} {localFile} {address}:{remoteFile}"
    print(command)
    os.system(command)

def ssh(address, command, pipeIn=None):
    global ZONE
    command = f"gcloud compute ssh --zone={ZONE} {address} --command='{command}'"
    if pipeIn:
        command = f"{pipeIn} | {command}"
    print(command)
    os.system(command)

def sendCheckpoint(address, checkpoint):
    print("Sending checkpoint:")
    scp(address, checkpoint, f"~/{os.path.split(checkpoint)[1]}")

def sendModel(address, modelpath):
    print(f"sending model {modelpath} to remote home directory")
    scp(address, modelpath, f"~") #TODO

def sendCommand(
        address,
        script,
        job_id,
        upload,
        ssh_key,
        shutdown):
    import tempfile

    br = branch()

    # dealing with checkpoints, perhaps remove
    copyCheckpoint = ""

    preamble = f"""#!/bin/bash
source ~/.bashrc
cd ~/ProgramSearch
{copyCheckpoint}
git fetch
git checkout {br}
git pull
"""
    #hack for non-kevin users ...
    #deal with this TODO
#     if user() != "ellisk":
#         cp_str = """#!/bin/bash
# cp -r ../ellisk/ec ~/ec
# """
#         preamble = cp_str + preamble

    preamble += "mv ~/patch ~/ProgramSearch/patch\n"
    preamble += "git apply patch ; mkdir jobs\n"

    if upload:
        # This is probably a terribly insecure idea...
        # But I'm not sure what the right way of doing it is
        # I'm just going to copy over the local SSH identity
        # Assuming that this is an authorized key at the upload site this will work
        # Of course it also means that if anyone were to pull the keys off of AWS,
        # they would then have access to every machine that you have access to
        UPLOADFREQUENCY = 60 * 3  # every 3 minutes
        uploadCommand = """\
rsync  -e 'ssh  -o StrictHostKeyChecking=no' -avz \
experimentOutputs checkpoints jobs {}""".format(upload) # TODO change folders that are synced
        preamble += """
mv ~/.ssh/%s ~/.ssh/id_rsa
mv ~/.ssh/%s.pub ~/.ssh/id_rsa.pub
chmod 600 ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa.pub
bash -c "while sleep %d; do %s; done" &> /tmp/test.txt & 
UPLOADPID=$!
""" % (ssh_key, ssh_key, UPLOADFREQUENCY, uploadCommand)
    
    script = preamble + script

    if upload:
        script += """
kill -9 $UPLOADPID
%s
""" % (uploadCommand)

    if shutdown:
        script += """
sudo shutdown -h now
"""

    fd = tempfile.NamedTemporaryFile(mode='w', delete=False, dir="/tmp")
    fd.write(script)
    fd.close()
    name = fd.name

    print("SCRIPT:")
    print(script)

    # Copy over the script
    print("Copying script over to", address)
    scp(address, name, "~/script.sh")

    # delete local copy
    os.system("rm %s" % name)

    # Send keys for openmind
    if upload:
        print("Uploading your ssh identity")
        scp(address, f"~/.ssh/{ssh_key}", f"~/.ssh/{ssh_key}")
        scp(address, f"~/.ssh/{ssh_key}.pub", f"~/.ssh/{ssh_key}.pub")

    # Send git patch
    print("Sending git patch over to", address)
    os.system("git diff --stat")
    ssh(address, "cat > ~/patch",
        pipeIn=f"""(echo "Base-Ref: $(git rev-parse origin/{br})" ; echo ; git diff --binary origin/{br})""")

    # Execute the script
    # For some reason you need to pipe the output to /dev/null in order to get
    # it to detach
    ssh(address, "bash ./script.sh > /dev/null 2>&1 &")
    print("Executing script on remote host.")


def launchExperiment(
        name,
        command,
        gpuType=None,
        gpuCount=None,
        copyModel=None,
        tail=False,
        upload=None,
        ssh_key="id_rsa",
        shutdown=True,
        size="f1-micro"):
    job_id = "{}_{}_{}".format(name, user(), datetime.now().strftime("%FT%T"))
    job_id = job_id.replace(":", ".")
    if upload is None and shutdown:
        print("You didn't specify an upload host, and also specify that the machine should shut down afterwards. These options are incompatible because this would mean that you couldn't get the experiment outputs.")
        sys.exit(1)

    # building script:
    #command = f"python dummy.py&&singularity exec --nv container.img {command}"
    script = """
%s > jobs/%s 2>&1
""" % (command, job_id)

    name = job_id
    instance, address = launchGoogleCloud(size, name, gpuType, gpuCount)
    time.sleep(180) #TODO deal with this ...
    if copyModel is not None:
        sendModel(address, copyModel)
    sendCommand(
        address,
        script,
        job_id,
        upload,
        ssh_key,
        shutdown)
    if tail: #TODO
        ssh(address, f""" \
                    mkdir -p ProgramSearch/jobs && \
                    touch ProgramSearch/jobs/{job_id} && \
                    tail -f -n+0 ProgramSearcj/jobs/{job_id} \
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-u', "--upload",
                        default={
                            "mnye": "mnye@openmind7.mit.edu:/om/user/mnye/REPLAbstraction",
                            "ellisk": "ellisk@openmind7.mit.edu:/om2/user/ellisk/ProgramSearch",
                        }.get(user(), None))
    parser.add_argument('-z', "--size",
                        default="f1-micro")
    parser.add_argument("--tail",
                        default=False,
                        help="attach to the machine and tail ec's output.",
                        action="store_true")
    parser.add_argument("--copy", type=str, default=None,
                        help="send model to copy over")
    #TODO advanced copy
    parser.add_argument('-k', "--shutdown",
                        default=False,
                        action="store_true")
    parser.add_argument('-g', "--gpuType", type=str, default=None, help="gpu type to use", choices=['t4','p100', 'k80'])
    parser.add_argument('--gpuCount', type=int, default=None)
    parser.add_argument("--ssh_key", default='id_rsa', help="Name of local RSA key file for openmind.")
    parser.add_argument("name")
    parser.add_argument("command")
    arguments = parser.parse_args()

    launchExperiment(arguments.name,
                     arguments.command,
                     gpuType=arguments.gpuType,
                     gpuCount=arguments.gpuCount,
                     shutdown=arguments.shutdown,
                     tail=arguments.tail,
                     copyModel=arguments.copy,
                     size=arguments.size,
                     upload=arguments.upload,
                     ssh_key=arguments.ssh_key)
