#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, datetime as dt, json, math, urllib.request
from pathlib import Path
import numpy as np
from numba import njit, prange, set_num_threads

INITIAL=100_000_000.0; COMM=.00015; TAX=.154; RESERVE=30_000_000.0
URL='https://raw.githubusercontent.com/kh-kim/stock_market_reinforcement_learning/master/sample_data/000660.csv'
LOOKBACKS=np.array([1,2,3,5,10,20,60],dtype=np.int16)


def load_rows(url):
    with urllib.request.urlopen(url,timeout=60) as r:text=r.read().decode()
    rows=[]
    for line in text.splitlines():
        p=line.strip().split(',')
        if len(p)>=5:rows.append((p[0],float(p[1]),float(p[2]),float(p[3]),float(p[4])))
    rows.sort();return rows

def prep(rows):
    n=len(rows);dates=[r[0] for r in rows];o=np.array([r[1] for r in rows]);h=np.array([r[2] for r in rows]);l=np.array([r[3] for r in rows]);c=np.array([r[4] for r in rows])
    seg=np.zeros(n,np.int16);sid=0;breaks=[]
    for i in range(1,n):
        gap=(dt.date.fromisoformat(dates[i])-dt.date.fromisoformat(dates[i-1])).days;rr=c[i]/c[i-1]-1
        if abs(rr)>.35 or gap>30:sid+=1;breaks.append({'date':dates[i],'gap':gap,'return':rr})
        seg[i]=sid
    ret=np.full((n,len(LOOKBACKS)),np.nan);valid=np.zeros(n,np.int8)
    for i in range(1,n):
        j=i-1
        if j<60 or seg[j]!=seg[j-60]:continue
        valid[i]=1
        for k,p in enumerate(LOOKBACKS):ret[i,k]=c[j]/c[j-p]-1
    prices=np.full((n,2,2),np.nan);prices[0,:,:]=10000.
    for i in range(1,n):
        if seg[i]!=seg[i-1]:prices[i,:,:]=10000.;continue
        for asset,sgn in ((0,2.),(1,-2.)):
            prev=prices[i-1,asset,1]
            for ph,x in enumerate((o[i],c[i])):
                factor=1+sgn*(x/c[i-1]-1)
                if factor>0:prices[i,asset,ph]=prev*factor
    return dates,seg,ret,valid,prices,breaks

def candidates():
    rows=[]
    for lb_i,p in enumerate(LOOKBACKS):
      for thr_i in range(0,31):
       thr=thr_i/100
       for ep in (0,1):
        for xp in (0,1):
         for hold in range(1,11):
          for funding in (0,1):rows.append((0,lb_i,thr,0.,ep,xp,hold,1,funding))
    lbset=[2,3,4,5,6];trends=[0.,.02,.05,.10,.15,.20];moves=[.02,.03,.04,.05,.07,.10]
    for kind in (1,2):
     for lb_i in lbset:
      for trend in trends:
       for move in moves:
        for ep in (0,1):
         for xp in (0,1):
          for lh in range(1,7):
           inv_holds=range(1,7) if kind==1 else (1,)
           for ih in inv_holds:
            for funding in (0,1):rows.append((kind,lb_i,trend,move,ep,xp,lh,ih,funding))
    return np.array(rows,np.float64)

@njit(cache=False)
def action_for(kind,lb_i,a,b,rrow,valid):
    if valid==0:return 0
    x=rrow[lb_i];r1=rrow[0]
    if kind==0:return 1 if x>=a else 0
    if x>=a and r1<=-b:return 1
    if kind==1 and x<=-a and r1>=b:return -1
    return 0

@njit(cache=False)
def sim_one(par,ret,valid,prices,seg,start,end,slip,detailed=False):
    kind=int(par[0]);lb=int(par[1]);a=par[2];b=par[3];ep=int(par[4]);xp=int(par[5]);lh=int(par[6]);ih=int(par[7]);funding=int(par[8])
    eq=INITIAL;peak=eq;mdd=0.;trades=longs=invs=0;i=start
    while i<=end:
        act=action_for(kind,lb,a,b,ret[i],valid[i])
        if act==0:i+=1;continue
        asset=0 if act==1 else 1;hold=lh if act==1 else ih;j=i+hold
        if j>end or seg[i]!=seg[j]:i+=1;continue
        pin=prices[i,asset,ep];pout=prices[j,asset,xp]
        if not np.isfinite(pin) or not np.isfinite(pout):i+=1;continue
        reserve=RESERVE if funding==0 else 0.;invest=max(0.,eq-reserve)
        if invest<=0:i+=1;continue
        q=invest/(pin*(1+COMM+slip));cash=eq-q*pin*(1+COMM+slip)
        if detailed:
            for k in range(i,j+1):
                for ph in range(2):
                    if k==i and ph<ep:continue
                    if k==j and ph>xp:continue
                    mark=prices[k,asset,ph]
                    if not np.isfinite(mark):continue
                    mf=mark*(1-slip);tax=max(0.,q*(mf-pin))*TAX;liq=cash+q*mf*(1-COMM)-tax
                    peak=max(peak,liq);mdd=min(mdd,liq/peak-1)
        gross=q*(pout-pin);eq=cash+q*pout*(1-COMM-slip)-max(gross,0.)*TAX
        peak=max(peak,eq);mdd=min(mdd,eq/peak-1);trades+=1
        if act==1:longs+=1
        else:invs+=1
        i=j+2 if funding==1 else j+(0 if xp<ep else 1)
    return eq/INITIAL-1,mdd,trades,longs,invs

@njit(parallel=True,cache=False)
def batch(params,ret,valid,prices,seg,blocks,slip):
    out=np.empty((len(params),len(blocks)*2),np.float64)
    for z in prange(len(params)):
        for w in range(len(blocks)):
            x=sim_one(params[z],ret,valid,prices,seg,blocks[w,0],blocks[w,1],slip,False)
            out[z,w]=x[0];out[z,len(blocks)+w]=x[2]
    return out

def describe(par):
    kind=int(par[0]);lb=int(LOOKBACKS[int(par[1])]);a=float(par[2]);b=float(par[3])
    return {'kind':['momentum_long','symmetric_trend_pullback','long_trend_pullback'][kind],'lookback':lb,'threshold_or_trend':a,'move':b,'entry_phase':['open','close'][int(par[4])],'exit_phase':['open','close'][int(par[5])],'long_hold':int(par[6]),'inverse_hold':int(par[7]),'funding':['reserve_30m','full_tplus2'][int(par[8])]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--top-csv',type=Path,required=True);ap.add_argument('--threads',type=int,default=4);args=ap.parse_args();set_num_threads(args.threads)
    rows=load_rows(URL);dates,seg,ret,valid,prices,breaks=prep(rows);pars=candidates();start=60;n=len(rows)-start;k=8
    blocks=np.array([(start+n*i//k,start+n*(i+1)//k-1) for i in range(k)],np.int32)
    base=batch(pars,ret,valid,prices,seg,blocks,.001);rets=base[:,:k];counts=base[:,k:]
    lengths=np.array([e-s+1 for s,e in blocks]);growth=np.log1p(np.maximum(rets,-.999999))/lengths
    ids=np.where((rets>0).all(1)&(counts>=3).all(1))[0]
    if len(ids)==0:
        pos=(rets>0).sum(1);mx=pos.max();ids=np.where(pos==mx)[0]
    order=ids[np.argsort(growth[ids].min(1))[::-1]];audited=[]
    for z in order[:1000]:
        full=sim_one(pars[z],ret,valid,prices,seg,start,len(rows)-1,.001,True)
        if full[1]<-.35:continue
        stressed=[sim_one(pars[z],ret,valid,prices,seg,s,e,.01,False)[0] for s,e in blocks]
        audited.append({'candidate_id':int(z),'rule':describe(pars[z]),'block_returns':[float(x) for x in rets[z]],'block_trades':[int(x) for x in counts[z]],'minimum_return':float(rets[z].min()),'minimum_per_session_growth':float(growth[z].min()),'full':{'return':float(full[0]),'mdd':float(full[1]),'trades':int(full[2]),'longs':int(full[3]),'inverses':int(full[4])},'stress_1pct_block_returns':stressed,'stress_1pct_all_positive':all(x>0 for x in stressed)})
    robust=[x for x in audited if x['stress_1pct_all_positive']]
    ranked=sorted(robust or audited,key=lambda x:(x['minimum_per_session_growth'],min(x['stress_1pct_block_returns']),x['full']['mdd']),reverse=True)
    report={'schema_version':1,'trade_ready':False,'classification':'old_public_history_training_only','source':{'url':URL,'rows':len(rows),'start':dates[0],'end':dates[-1],'breaks':breaks},'grid_candidates':len(pars),'old_blocks':[{'name':f'O{i+1}','start':dates[s],'end':dates[e],'sessions':e-s+1} for i,(s,e) in enumerate(blocks)],'all_blocks_positive_min3_count':int(((rets>0).all(1)&(counts>=3).all(1)).sum()),'audited_count':len(audited),'stress_survivors':len(robust),'selected':ranked[0] if ranked else None,'top100':ranked[:100],'notes':['selection uses only 2003-2016 public history; 2024-2026 and actual ETF data are reserved for later holdout evaluation','daily-reset synthetic +/-2x proxy; actual products did not exist','large discontinuities and >30-day gaps reset proxy and require 60-session warmup']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    args.top_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.top_csv.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['candidate_id','rule_json','min_return','full_return','full_mdd','trades','longs','inverses','stress_all_positive','block_returns','stress_returns'])
        for x in ranked[:100]:w.writerow([x['candidate_id'],json.dumps(x['rule'],separators=(',',':')),x['minimum_return'],x['full']['return'],x['full']['mdd'],x['full']['trades'],x['full']['longs'],x['full']['inverses'],x['stress_1pct_all_positive'],json.dumps(x['block_returns']),json.dumps(x['stress_1pct_block_returns'])])
    print(json.dumps({'grid':len(pars),'positive':report['all_blocks_positive_min3_count'],'stress_survivors':len(robust),'selected':report['selected']},ensure_ascii=False))
if __name__=='__main__':main()
