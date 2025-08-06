import subprocess
import re
import threading
import sys

def read_output(proc):
    url = None
    GREEN = '\033[32m'
    RESET = '\033[0m'
    for line in proc.stdout:
        print(line, end='')
        m = re.search(r'https://[^\s]+\.trycloudflare\.com', line)
        if m and url is None:
            url = m.group(0)
            print(f'检测到公网地址: {GREEN}{url}{RESET}')
    return url


def start_cloudflared_and_get_url(port):
    proc = subprocess.Popen(
        ['cloudflared', 'tunnel', '--url', f'http://localhost:{port}'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    thread = threading.Thread(target=read_output, args=(proc,), daemon=True)
    thread.start()

    return proc

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python tunnel.py <端口号>")
        sys.exit(1)

    port = sys.argv[1]
    proc = start_cloudflared_and_get_url(port)
    print(f'cloudflared 隧道已启动（端口 {port}），进程ID:', proc.pid)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print('退出，关闭隧道进程')
        proc.terminate()
