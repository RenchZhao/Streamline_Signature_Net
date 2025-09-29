import subprocess


def run_command(command, shell=True, log=None):
    if not log:
        try:
            # print command
            if isinstance(command,list):
                print(' '.join(command))
            else:
                print(command)
            
            # run command
            if shell:
                result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # print stdout
            stdout = result.stdout
            if isinstance(stdout,bytes):
                stdout = stdout.decode()
            print(stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"error occurred while running the command:{e}")
            if e.output:
                print(f"Output:{e.output.decode()}")
            if e.stderr:
                print(f"stderr:{e.stderr.decode()}")
            return
    else:
        try:
            # print command
            if isinstance(command,list):
                log.record_print(' '.join(command))
            else:
                log.record_print(command)

            # run command
            if shell:
                result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # print stdout
            stdout = result.stdout
            if isinstance(stdout,bytes):
                stdout = stdout.decode()
            log.record_print(stdout)
            return result
        except subprocess.CalledProcessError as e:
            log.record_print(f"error occurred while running the command:{e}")
            if e.output:
                log.record_print(f"Output:{e.output.decode()}")
            if e.stderr:
                log.record_print(f"stderr:{e.stderr.decode()}")
            return