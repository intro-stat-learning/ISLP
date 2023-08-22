'''
Run notebooks in an isolated environment specified by a requirements.txt file
'''

from hashlib import md5
import tempfile
import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--requirements',
                    default='requirements.txt')
parser.add_argument('labs',
                    metavar='N',
                    type=str,
                    nargs='+')
parser.add_argument('--python',
                    default='3.10')
parser.add_argument('--tarball',
                    default=None,
                    dest='tarball')
parser.add_argument('--inplace',
                    default=False,
                    action='store_true',
                    help='run notebooks in place?')
parser.add_argument('--timeout',
                    default=5000,
                    help='preprocessor timeout')

def make_notebooks(requirements='requirements.txt',
                   srcs=[],
                   dests=[],
                   tarball='',
                   inplace=False,
                   tmpdir='',
                   python='3.10',
                   timeout=5000 # should be enough for Ch10
                   ):

    if tarball and inplace:
        raise ValueError('tarball option expects notebooks in a tmpdir, while inplace does not copy to a tmpdir')
    
    md5_ = md5()
    md5_.update(open(requirements, 'rb').read());
    hash_ = md5_.hexdigest()[:8]

    env_name = f'isolated_env_{hash_}'

    setup_cmd = f'''
    conda create -n {env_name} python={python} -y;
    conda run -n {env_name} pip install -r {requirements} jupyter jupytext;
    '''

    print(setup_cmd)
    os.system(setup_cmd)

    archive_files = []
    for src_, dest_ in zip(srcs, dests):
        if src_ != dest_:
            os.system(f'cp {src_} {dest_}')
        name = os.path.split(dest_)[1]
        build_cmd = f'''conda run -n {env_name} jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout={timeout} {dest_} '''
        if '02' in name:
            build_cmd += ' --allow-errors '

        print(build_cmd)
        os.system(build_cmd)
        archive_files.append(name)

    archive_files = ' '.join(archive_files)

    if tarball:
        tarball = os.path.abspath(tarball)
        tarball_cmd = f'''
        cd {tmpdir}; tar -cvzf {tarball} {archive_files}
        '''
        print(tarball_cmd)
        os.system(tarball_cmd)

    os.system(f'conda env remove -n {env_name}')

if __name__ == '__main__':

    args = parser.parse_args()
    srcs = [os.path.abspath(l) for l in args.labs]

    tmpdir = tempfile.mkdtemp()

    if args.inplace:
        dests = srcs
    else:
        dests = [os.path.join(tmpdir, os.path.split(l)[1]) for l in args.labs]

    make_notebooks(requirements=os.path.abspath(args.requirements),
                   srcs=srcs,
                   dests=dests,
                   inplace=args.inplace,
                   tmpdir=tmpdir,
                   python=args.python,
                   tarball=args.tarball,
                   timeout=args.timeout)
