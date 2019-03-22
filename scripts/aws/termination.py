import urllib
from urllib.request import urlopen
import logging
import subprocess
import time

TERMINATION_URL = 'http://169.254.169.254/latest/meta-data/spot/termination-time'
POLL_INTERVAL = 5

def run():
    logging.info('Starting.')
    not_terminated = True
    while not_terminated:
        try:
            time.sleep(POLL_INTERVAL)
            req = urlopen(TERMINATION_URL)
            not_terminated = False
        except urllib.error.HTTPError as e:
            if e.getcode() != 404:
                logging.error('Unexpected response code ', e)
        except urllib.error.URLError as e:
            logging.error('Unexpected error ', e)
    logging.info('Received termination notice!')
    logging.info('Scheduled to terminate at ', req.read())
    logging.info('Shutting down Ray cleanly.')
    subprocess.check_call(['ray', 'stop'])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
