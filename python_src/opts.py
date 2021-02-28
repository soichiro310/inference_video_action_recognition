import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--host',
                        type=str,
                        help='select host(ip address) of execute app'
                       )
    
    parser.add_argument('--port',
                       type=int,
                       default=8888
                       )
    
    parser.add_argument('--sample_video_dir',
                       type=str,
                       default='./sample_video'
                       )
    
    return parser


# argparse debug 
if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)