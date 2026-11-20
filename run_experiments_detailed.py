"""Run the exact, MINLP, and sparse domain-reduction experiments."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
THREAD_ENV = ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
J_NONSPARSE = [10, 20, 100, 1000, 10000, 100000]
J_SPARSE = [10, 20, 100, 1000, 10000, 100000]
BENCH_MAPPING = dict(minlp_status='status', minlp_incumbent_SSE='incumbent_SSE',
                     minlp_best_bound_SSE='best_bound_SSE', minlp_gap_percent='mip_gap_percent',
                     minlp_solver_runtime_seconds='solver_runtime_seconds',
                     minlp_build_runtime_seconds='build_runtime_seconds',
                     minlp_total_runtime_seconds='total_runtime_seconds', minlp_solution_count='solution_count',
                     minlp_node_count='node_count', minlp_breakpoint='breakpoint', minlp_selected_count='selected_count')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    datasets = ['simulated', *json.loads((ROOT / 'data/sources.json').read_text())]
    p.add_argument('--datasets', nargs='+', choices=datasets, default=datasets)
    p.add_argument('--settings', nargs='+', choices=['nonsparse','sparse'], default=['nonsparse','sparse'])
    p.add_argument('--n-seeds', type=int, default=1, help='Simulation replications; real datasets run once')
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--j-nonsparse', nargs='+', type=int, default=J_NONSPARSE)
    p.add_argument('--j-sparse', nargs='+', type=int, default=J_SPARSE)
    p.add_argument('--T', type=int, default=100, help='Simulation only')
    p.add_argument('--sparse-fraction', type=float, default=0.2)
    p.add_argument('--detail-max-J', type=int, default=1100, help='Complete NPZ fits saved at every J')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/raw')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--threads', type=int, default=1, help='Numba and numerical library threads (default: 1)')
    p.add_argument('--minlp-time-limit', type=float, default=10000.0)
    p.add_argument('--minlp-threads', type=int, default=1)
    p.add_argument('--minlp-max-J', type=int, default=0, help='Optional explicit cap; 0=none')
    p.add_argument('--skip-minlp', action='store_true')
    domain_mode = p.add_mutually_exclusive_group()
    domain_mode.add_argument('--domain-reduction-only', action='store_true',
                             help='Run sparse domain reduction on all datasets by default')
    domain_mode.add_argument('--skip-domain-reduction', action='store_true',
                             help='Run the original exact and MINLP experiment matrix')
    p.add_argument('--minlp-log', action='store_true', help='Also print solver log; log files always saved')
    p.add_argument('--skip-figures', action='store_true')
    p.add_argument('--fail-fast', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume same configuration in --output-dir')
    p.add_argument('--smoke', action='store_true', help='Simulated J=5,T=10 and 2-second MINLP limits')
    args=p.parse_args(argv)
    if args.domain_reduction_only:
        args.settings, args.skip_minlp, args.skip_figures = ['sparse'], True, True
    if args.smoke:
        args.datasets,args.j_nonsparse,args.j_sparse=['simulated'],[5],[5]
        args.T,args.n_seeds,args.minlp_time_limit,args.skip_figures=10,1,2.0,True
    if args.n_seeds<1 or args.seed_start<0 or args.T<6 or any(j<1 for j in args.j_nonsparse+args.j_sparse):
        p.error('Require n-seeds>=1, seed-start>=0, T>=6, J>=1')
    if not 0<=args.sparse_fraction<=1 or min(args.threads,args.minlp_threads)<1 or args.minlp_time_limit<=0:
        p.error('Invalid fraction, threads or time limit')
    if min(args.minlp_max_J,args.detail_max_J)<0:
        p.error('Dimension caps must be nonnegative')
    if args.resume and args.output_dir is None:
        p.error('--resume requires --output-dir')
    if args.output_dir is None:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        args.output_dir=ROOT/'experiment_results_detailed/runs'/stamp
    return args


def environment():
    packages={}
    for name in ['numpy','pandas','numba','openpyxl','gurobipy','matplotlib','threadpoolctl']:
        try:
            packages[name]=version(name)
        except PackageNotFoundError:
            packages[name]=None
    return dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
                cpu_count=os.cpu_count(),packages=packages,
                thread_environment={k:os.environ.get(k) for k in THREAD_ENV})


def snapshot(out,rows,args,warmups):
    from reporting import FIELDS, write_instance_workbook, write_final_summary

    temp=out/'all_replications.tmp.csv'
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temp,out/'all_replications.csv')
    for dataset,setting,J in sorted({(r['dataset'],r['setting'],int(r['J'])) for r in rows}):
        sub=[r for r in rows if (r['dataset'],r['setting'],int(r['J']))==(dataset,setting,J)]
        write_instance_workbook(out/'runtime_workbooks'/dataset/f'{setting}_J_{J}_all_seeds.xlsx',sub,warmups)
    temp=out/'final_summary.tmp.xlsx'
    write_final_summary(temp,rows,args,warmups)
    os.replace(temp,out/'final_summary.xlsx')


def cases(args):
    from datasets import Dataset, load_dataset

    for name in args.datasets:
        if name=='simulated':
            for setting in args.settings:
                for J in (args.j_nonsparse if setting=='nonsparse' else args.j_sparse):
                    for seed in range(args.seed_start,args.seed_start+args.n_seeds):
                        yield name,setting,J,seed,None
        else:
            started=time.perf_counter()
            try:
                loaded=load_dataset(name,args.data_dir,args.offline)
                loaded.metadata['shared_dataset_loading_seconds']=time.perf_counter()-started
            except Exception as exc:
                loaded=exc
            for setting in args.settings:
                yield name,setting,len(loaded.Y) if isinstance(loaded,Dataset) else 0,0,loaded


def direct_residual_sse(data, result):
    import numpy as np

    hinge = np.maximum(0, data.x - result['shared_breakpoint'])
    direct = 0.
    for a in range(0, len(data.Y), 1024):
        block = slice(a, a + 1024)
        pred = (result['alpha'][block, None] + result['left_slope'][block, None] * data.x
                + result['delta'][block, None] * hinge)
        direct += float(np.sum((data.Y[block] - pred)**2))
    return direct


def main(argv=None):
    args=parse_args(argv)
    # In the CLI, set limits before importing NumPy or initializing Numba.
    if __name__ == '__main__':
        for name in THREAD_ENV:
            os.environ[name] = str(args.threads)
    import numpy as np
    from datasets import Dataset
    from data_simulator import generate_synthetic_stock_data
    from domain_reduction import warm_up_domain_reduction
    from fitting import solve_domain_reduction, solve_exact
    from minlp_benchmark import parameter_bounds, exact_bound_diagnostics
    from minlp_worker import run_isolated
    from reporting import FIELDS, TIME_FIELDS, DOMAIN_CHECKPOINT_METRICS, save_detail_excel
    from run_optimization_nonsparse import warm_up_exact_numba
    from run_optimization_sparse import warm_up_numba_sparse_exact
    from numba import get_num_threads, set_num_threads
    from threadpoolctl import threadpool_info, threadpool_limits

    args.output_dir=args.output_dir.resolve()
    args.data_dir=args.data_dir.resolve()
    set_num_threads(args.threads)
    threadpool_limits(limits=args.threads)
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    cfgpath=out/'configuration.json'; rows=[]
    if cfgpath.exists():
        if not args.resume:
            raise FileExistsError(f'{out} already contains a run; use a new folder or --resume')
        recorded=json.loads(cfgpath.read_text())
        recorded.setdefault('domain_reduction_only', False)
        recorded.setdefault('skip_domain_reduction', True)
        for field in ['data_dir','output_dir']:
            recorded[field]=str(Path(recorded[field]).resolve())
        if recorded!=config:
            raise ValueError('Resume configuration differs from recorded run. Use --skip-domain-reduction for runs created before domain reduction was added.')
        csv_path=out/'all_replications.csv'
        if csv_path.exists():
            with csv_path.open(newline='') as f:
                rows=[dict.fromkeys(FIELDS,'') | {k:v for k,v in r.items() if k in FIELDS}
                      for r in csv.DictReader(f)]
    cfgpath.write_text(json.dumps(config,indent=2)+'\n')
    env = environment()
    env.update(numba_threads=get_num_threads(), native_threadpools=threadpool_info())
    (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    print(f'Results: {out}',flush=True)
    if not args.skip_minlp:
        print(f'MINLP limit={args.minlp_time_limit:g}s; J cap={args.minlp_max_J} (0=none)',flush=True)
    warmups={}
    if not args.domain_reduction_only:
        for setting, fn in [('nonsparse', warm_up_exact_numba), ('sparse', warm_up_numba_sparse_exact)]:
            if setting in args.settings:
                t0=time.perf_counter(); fn(); warmups[setting]=time.perf_counter()-t0
    if 'sparse' in args.settings and not args.skip_domain_reduction:
        t0=time.perf_counter()
        warm_up_domain_reduction()
        warmups['domain_reduction']=time.perf_counter()-t0
    terminal={'OPTIMAL','TIME_LIMIT','INFEASIBLE','SKIPPED_FLAG','SKIPPED_MAX_J'}
    for name,setting,J,seed,loaded in cases(args):
        run_domain = setting == 'sparse' and not args.skip_domain_reduction
        key=(name,setting,J,seed)
        def same(r):
            return (r['dataset'],r['setting'],int(r['J']),int(r['seed']))==key
        if any(same(r) and r['status']=='ok' and r['minlp_status'] in terminal
               and (not run_domain or r['domain_status']=='OK') for r in rows):
            print(f'Resume: already complete {key}',flush=True); continue
        rows=[r for r in rows if not same(r)]
        r=dict.fromkeys(FIELDS,'')
        r.update(dataset=name,setting=setting,J=J,T=args.T if name=='simulated' else len(loaded.x) if isinstance(loaded,Dataset) else 0,
                 seed=seed,k=math.floor(args.sparse_fraction*J) if setting=='sparse' else J)
        r.update({field:0.0 for field in TIME_FIELDS}); rows.append(r)
        if args.domain_reduction_only:
            for field in ('exact_end_to_end_seconds', 'numerical_runtime_seconds', 'statistics_seconds',
                          'interval_solver_seconds', 'selection_seconds', 'recovery_seconds'):
                r[field] = ''
            r['minlp_status'] = 'SKIPPED_FLAG'
        started=time.perf_counter(); failure=None
        print(f'Starting {name} {setting} J={J} seed={seed}',flush=True)
        try:
            t0=time.perf_counter()
            if isinstance(loaded,Exception):
                raise loaded
            if name=='simulated':
                x,Y,dates,names,_,_=generate_synthetic_stock_data(D=J,T=args.T,random_seed=seed)
                data=Dataset(name,x,Y,names,{'dates':list(map(str,dates)),'seed':seed}).validate()
            else:
                data=loaded
                r['data_loading_seconds']=data.metadata['shared_dataset_loading_seconds']
            r['data_generation_seconds']=time.perf_counter()-t0
            meta=out/'datasets'/f'{name}_J_{J}_seed_{seed:06d}.json'
            meta.parent.mkdir(parents=True,exist_ok=True); meta.write_text(json.dumps(data.metadata,indent=2)+'\n')
            digest=hashlib.sha256(data.x.tobytes()); digest.update(memoryview(data.Y))
            r['input_sha256']=digest.hexdigest()
            stem=Path(name)/setting/f'J_{J}'/f'seed_{seed:06d}'
            if run_domain:
                try:
                    domain = solve_domain_reduction(data.x, data.Y, k=r['k'])
                    t0 = time.perf_counter()
                    domain_direct = direct_residual_sse(data, domain)
                    if not np.isclose(domain_direct, domain['feasible_loss'], rtol=1e-6, atol=1e-6):
                        raise ArithmeticError('Recovered domain reduction SSE differs from solver SSE')
                    r['domain_validation_seconds'] = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    path = (out/'domain_reduction_fits'/stem).with_suffix('.npz')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, x=data.x, Y=data.Y, names=np.asarray(data.names, dtype=str), **domain)
                    r['domain_npz_writing_seconds'] = time.perf_counter() - t0
                    if J <= min(args.detail_max_J, 16382):
                        t0 = time.perf_counter()
                        save_detail_excel((out/'domain_reduction_detailed_fits'/stem).with_suffix('.xlsx'),
                                          data.x, data.Y, data.metadata.get('dates', data.x), data.names,
                                          domain, setting, seed, r['k'])
                        r['domain_excel_writing_seconds'] = time.perf_counter() - t0
                    r.update(domain_status='OK', domain_breakpoint=domain['shared_breakpoint'],
                             domain_feasible_SSE=domain['feasible_loss'], domain_direct_residual_SSE=domain_direct,
                             domain_numerical_runtime_seconds=domain['solve_runtime'],
                             domain_end_to_end_seconds=domain['end_to_end_seconds'])
                    for i, depth in enumerate(domain['refinement_iterations']):
                        for metric in DOMAIN_CHECKPOINT_METRICS:
                            r[f'domain_{depth}_{metric}'] = domain[f'refinement_{metric}'][i]
                    if args.domain_reduction_only:
                        r.update(status='ok', baseline_SSE=domain['C'])
                    del domain
                except Exception as exc:
                    r.update(domain_status='ERROR', domain_error=f'{type(exc).__name__}: {exc}')
                    if args.domain_reduction_only or args.fail_fast:
                        raise
                if args.domain_reduction_only:
                    continue
            result=solve_exact(data.x,data.Y,k=r['k'] if setting=='sparse' else None)
            r.update(status='ok',breakpoint=result['shared_breakpoint'],optimal_SSE=result['optimal_loss'],baseline_SSE=result['C'],
                     statistics_seconds=result.get('statistics_runtime',result.get('baseline_runtime',0)+result.get('prefix_runtime',0)),
                     interval_solver_seconds=result.get('interval_solve_runtime',result.get('interval_kernel_runtime',0)),
                     selection_seconds=result.get('selection_runtime',0),recovery_seconds=result['recovery_runtime'],
                     numerical_runtime_seconds=result['solve_runtime'],exact_end_to_end_seconds=result['end_to_end_seconds'])
            t0=time.perf_counter(); direct=direct_residual_sse(data,result)
            r.update(direct_residual_SSE=direct,objective_validation_abs_error=abs(direct-result['optimal_loss']))
            r['validation_seconds']=time.perf_counter()-t0
            if not np.isclose(direct,result['optimal_loss'],rtol=1e-6,atol=1e-6):
                raise ArithmeticError('Recovered SSE differs from optimized SSE')
            t0=time.perf_counter(); path=(out/'fits'/stem).with_suffix('.npz'); path.parent.mkdir(parents=True,exist_ok=True)
            arrays={k:v for k,v in result.items() if isinstance(v,(np.ndarray,float,int,bool))}
            np.savez_compressed(path,x=data.x,Y=data.Y,names=np.asarray(data.names,dtype=str),**arrays)
            r['data_npz_writing_seconds']=time.perf_counter()-t0
            if J<=min(args.detail_max_J,16382):
                t0=time.perf_counter()
                save_detail_excel((out/'detailed_fits'/stem).with_suffix('.xlsx'),data.x,data.Y,
                                  data.metadata.get('dates',data.x),data.names,result,setting,seed,r['k'])
                r['excel_writing_seconds']=time.perf_counter()-t0
            r['minlp_status']='PENDING'; snapshot(out,rows,args,warmups)
            if args.skip_minlp:
                r['minlp_status']='SKIPPED_FLAG'
            elif args.minlp_max_J and J>args.minlp_max_J:
                r['minlp_status']='SKIPPED_MAX_J'
            else:
                try:
                    t0=time.perf_counter(); bounds=parameter_bounds(data.x,data.Y)
                    r['bound_construction_seconds']=time.perf_counter()-t0
                    r.update(exact_bound_diagnostics(result,bounds))
                    path=(out/'bounds'/stem).with_suffix('.npz'); path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,**bounds)
                    del bounds
                    print(f'MINLP {name} {setting} J={J}: limit={args.minlp_time_limit:g}s (construction excluded)',flush=True)
                    bench=run_isolated(input_path=(out/'fits'/stem).with_suffix('.npz'),bounds_path=path,
                                       result_path=(out/'minlp_results'/stem).with_suffix('.json'),
                                       solution_path=(out/'minlp_fits'/stem).with_suffix('.npz'),
                                       log_path=(out/'minlp_logs'/stem).with_suffix('.log'),
                                       k=r['k'],time_limit=args.minlp_time_limit,threads=args.minlp_threads,
                                       seed=seed,console_log=args.minlp_log)
                    for dest,source in BENCH_MAPPING.items():
                        r[dest]=bench[source] if bench[source] is not None else ''
                    for dest,source in [('exact_minus_minlp_incumbent_SSE','incumbent_SSE'),('exact_minus_minlp_bound_SSE','best_bound_SSE')]:
                        if bench[source] is not None:
                            r[dest]=result['optimal_loss']-bench[source]
                    r['minlp_total_runtime_seconds']=bench['process_wall_seconds']+r['bound_construction_seconds']
                except Exception as exc:
                    r.update(minlp_status='ERROR',minlp_error=f'{type(exc).__name__}: {exc}')
                    if args.fail_fast:
                        failure=exc
        except Exception as exc:
            r.update(status='error',error=f'{type(exc).__name__}: {exc}')
            if run_domain and not r['domain_status']:
                r.update(domain_status='ERROR', domain_error=r['error'])
            if args.fail_fast:
                failure=exc
        finally:
            r['wall_clock_seconds']=time.perf_counter()-started
            snapshot(out,rows,args,warmups)
            if args.domain_reduction_only:
                print(f"Done {name} {setting} J={J}: domain reduction={r['domain_status']} {r['error']}", flush=True)
            else:
                print(f"Done {name} {setting} J={J}: exact={r['status']} MINLP={r['minlp_status']} "
                      f"domain reduction={r['domain_status']} {r['error']} {r['minlp_error']} {r['domain_error']}",flush=True)
        if failure is not None:
            raise failure
    if not args.skip_figures:
        from make_figures import make_all_figures
        make_all_figures(args.data_dir,out/'figures',args.offline)
    print(f'Completed {len(rows)} cases. Summary: {out / "final_summary.xlsx"}',flush=True)


if __name__=='__main__':
    main()
