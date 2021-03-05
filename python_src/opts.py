import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug',
                        action='store_true',
                        default=False
                       )

    parser.add_argument('--host',
                        type=str,
                        help='select host(ip address) of execute app',
                        default='127.0.0.1'
                       )
    
    parser.add_argument('--port',
                       type=int,
                       default=8888
                       )
    
    parser.add_argument('--sample_video_dir',
                       type=str,
                       default='./sample_video'
                       )
    
    parser.add_argument('--device',
                       type=str,
                       default='cpu',
                       help="'select device (cpu -> 'cpu', gpu -> 'cuda:0')"
                       )
    
    return parser

# argparse debug 
if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)