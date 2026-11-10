using System;

// Feasible SQP for a smooth objective with box constraints. Each iteration
// solves a box-constrained QP using a Powell-damped BFGS Hessian, followed by
// Armijo line search. No gradient-descent fallback is used in SQP mode.
public static class FinsRovSqpSolver
{
    public sealed class Result {
        public double Cost, ProjectedGradient, QpResidual;
        public int Iterations, QpSweeps, Evaluations, InvalidEvaluations, HessianRepairs, QpRetries;
        public bool Valid;
        public string Status;
    }
    public static Result Solve(double[,] sequence, Func<double[,],double[,],double> evaluate,
                               int maxIterations=32,double[,] initialHessian=null,bool activeSetQp=false)
    {
        int horizon=sequence.GetLength(0),axes=sequence.GetLength(1),n=horizon*axes;
        var gradient=new double[horizon,axes];double cost=evaluate(sequence,gradient);
        var result=new Result { Evaluations=1, Status="iteration_limit", Cost=cost,
            ProjectedGradient=double.PositiveInfinity, QpResidual=double.PositiveInfinity };
        // An infeasible rollout has no derivative, even when its legacy callback
        // fills the gradient with zeros. Never test stationarity on that value.
        if(!ValidEvaluation(cost,gradient) || !Bounded(sequence)) {
            result.Status="invalid_initial_prediction";result.InvalidEvaluations=1;return result;
        }
        result.Valid=true;
        var hessian=new double[n,n];
        for(int a=0;a<axes;a++) {
            double scale=1;
            for(int k=0;k<horizon;k++) scale=Math.Max(scale,Math.Abs(gradient[k,a]));
            for(int k=0;k<horizon;k++)hessian[k*axes+a,k*axes+a]=2.5*scale;
        }
        if(initialHessian!=null)Array.Copy(initialHessian,hessian,hessian.Length);
        for(int iteration=0;iteration<maxIterations;iteration++) {
            double[] x=Flatten(sequence),g=Flatten(gradient);
            result.ProjectedGradient=Stationarity(x,g);
            if(result.ProjectedGradient<1e-5) { result.Status="stationary";break; }
            var d=new double[n];var residual=(double[])g.Clone();
            if(activeSetQp) {
                var lower=new double[n];var upper=new double[n];for(int i=0;i<n;i++){lower[i]=-1-x[i];upper[i]=1-x[i];}
                int qpSteps;bool solved=TryActiveSetBoxQp(hessian,g,lower,upper,out d,out qpSteps);result.QpSweeps+=qpSteps;
                // BFGS/Gauss-Newton can lose positive definiteness in finite
                // precision. Repair only on failure; never alter the objective.
                if(!solved) {
                    RepairHessian(hessian,g,false);result.HessianRepairs++;result.QpRetries++;
                    solved=TryActiveSetBoxQp(hessian,g,lower,upper,out d,out qpSteps);result.QpSweeps+=qpSteps;
                }
                if(!solved) {
                    RepairHessian(hessian,g,true);result.HessianRepairs++;result.QpRetries++;
                    solved=TryActiveSetBoxQp(hessian,g,lower,upper,out d,out qpSteps);result.QpSweeps+=qpSteps;
                }
                if(!solved) {result.Status="qp_numerical_failure";break;}
                double qp=0;
                for(int i=0;i<n;i++){double r=g[i];for(int j=0;j<n;j++)r+=hessian[i,j]*d[j];qp=Math.Max(qp,Math.Abs(d[i]-Math.Max(lower[i],Math.Min(upper[i],d[i]-r))));}
                result.QpResidual=qp;
            }
            // Coordinate minimization of the convex QP: g'd + .5 d'Hd,
            // subject to -1-x <= d <= 1-x. Hessian remains positive definite.
            for(int sweep=0;!activeSetQp && sweep<128;sweep++) {
                for(int i=0;i<n;i++) {
                    double next=Math.Max(-1-x[i],Math.Min(1-x[i],d[i]-residual[i]/hessian[i,i]));
                    double delta=next-d[i];d[i]=next;
                    for(int j=0;j<n;j++)residual[j]+=hessian[j,i]*delta;
                }
                result.QpSweeps++;
                double qp=0;
                for(int i=0;i<n;i++)qp=Math.Max(qp,Math.Abs(d[i]-Math.Max(-1-x[i],Math.Min(1-x[i],d[i]-residual[i]))));
                result.QpResidual=qp;
                if(qp<1e-7)break;
            }
            double slope=0;for(int i=0;i<n;i++)slope+=g[i]*d[i];
            if(!Finite(slope) || slope>=-1e-12) { result.Status="no_descent_direction";break; }
            double alpha=1,nextCost=cost;bool accepted=false;var candidate=new double[horizon,axes];
            for(int search=0;search<16;search++) {
                for(int i=0;i<n;i++)candidate[i/axes,i%axes]=Math.Max(-1,Math.Min(1,x[i]+alpha*d[i]));
                nextCost=evaluate(candidate,null);result.Evaluations++;
                if(!ValidCost(nextCost))result.InvalidEvaluations++;
                if(ValidCost(nextCost) && nextCost<=cost+1e-4*alpha*slope) { accepted=true;break; }
                alpha*=.5;
            }
            if(!accepted) { result.Status="line_search_failed";break; }
            var nextGradient=new double[horizon,axes];nextCost=evaluate(candidate,nextGradient);result.Evaluations++;
            if(!ValidEvaluation(nextCost,nextGradient)) {
                result.InvalidEvaluations++;result.Status="invalid_candidate_derivative";break;
            }
            var s=new double[n];var y=new double[n];var hs=new double[n];
            for(int i=0;i<n;i++) {s[i]=candidate[i/axes,i%axes]-x[i];y[i]=nextGradient[i/axes,i%axes]-g[i];}
            double shs=0,sy=0;
            for(int i=0;i<n;i++) {
                for(int j=0;j<n;j++)hs[i]+=hessian[i,j]*s[j];
                shs+=s[i]*hs[i];sy+=s[i]*y[i];
            }
            if(Finite(shs) && Finite(sy) && shs>1e-14) {
                double theta=sy>=.2*shs?1:.8*shs/(shs-sy),sr=0;
                for(int i=0;i<n;i++) {y[i]=theta*y[i]+(1-theta)*hs[i];sr+=s[i]*y[i];}
                if(Finite(sr) && sr>1e-14)
                    for(int i=0;i<n;i++)for(int j=0;j<n;j++)hessian[i,j]+=y[i]*y[j]/sr-hs[i]*hs[j]/shs;
            }
            Array.Copy(candidate,sequence,sequence.Length);gradient=nextGradient;cost=nextCost;
            result.Iterations=iteration+1;
        }
        result.Cost=cost;result.ProjectedGradient=Stationarity(Flatten(sequence),Flatten(gradient));
        if(result.ProjectedGradient<1e-5)result.Status="stationary";
        return result;
    }
    static bool TryActiveSetBoxQp(double[,] h,double[] g,double[] lower,double[] upper,out double[] x,out int steps)
    {
        int n=g.Length;x=new double[n];var active=new int[n];steps=0;
        double scale=1;for(int i=0;i<n;i++)for(int j=0;j<n;j++) {
            if(!Finite(h[i,j]))return false;
            scale=Math.Max(scale,Math.Abs(h[i,j]));
        }
        for(int i=0;i<n;i++){x[i]=Math.Max(lower[i],Math.Min(upper[i],0));if(x[i]<=lower[i]+1e-10)active[i]=-1;else if(x[i]>=upper[i]-1e-10)active[i]=1;}
        for(int iteration=0;iteration<4*n+32;iteration++) {
            steps++;var r=(double[])g.Clone();for(int i=0;i<n;i++)for(int j=0;j<n;j++)r[i]+=h[i,j]*x[j];
            var free=new System.Collections.Generic.List<int>();for(int i=0;i<n;i++)if(active[i]==0)free.Add(i);
            int count=free.Count;var l=new double[count,count];var rhs=new double[count];
            for(int i=0;i<count;i++) {
                rhs[i]=-r[free[i]];
                for(int j=0;j<=i;j++) {
                    double sum=h[free[i],free[j]];for(int k=0;k<j;k++)sum-=l[i,k]*l[j,k];
                    if(!Finite(sum))return false;
                    if(i==j){if(sum<=1e-14*scale)return false;l[i,j]=Math.Sqrt(sum);}else l[i,j]=sum/l[j,j];
                }
            }
            for(int i=0;i<count;i++){for(int j=0;j<i;j++)rhs[i]-=l[i,j]*rhs[j];rhs[i]/=l[i,i];}
            for(int i=count-1;i>=0;i--){for(int j=i+1;j<count;j++)rhs[i]-=l[j,i]*rhs[j];rhs[i]/=l[i,i];}
            var direction=new double[n];double largest=0;for(int i=0;i<count;i++){if(!Finite(rhs[i]))return false;direction[free[i]]=rhs[i];largest=Math.Max(largest,Math.Abs(rhs[i]));}
            if(largest<1e-9) {
                int release=-1;double violation=1e-8;
                for(int i=0;i<n;i++){double value=active[i]<0?-r[i]:active[i]>0?r[i]:0;if(value>violation){violation=value;release=i;}}
                if(release<0)break;active[release]=0;continue;
            }
            double alpha=1;
            foreach(int i in free)if(direction[i]>1e-12)alpha=Math.Min(alpha,(upper[i]-x[i])/direction[i]);else if(direction[i]<-1e-12)alpha=Math.Min(alpha,(lower[i]-x[i])/direction[i]);
            for(int i=0;i<n;i++)x[i]=Math.Max(lower[i],Math.Min(upper[i],x[i]+alpha*direction[i]));
            if(alpha<1-1e-12)foreach(int i in free)if(x[i]<=lower[i]+1e-9)active[i]=-1;else if(x[i]>=upper[i]-1e-9)active[i]=1;
        }
        for(int i=0;i<n;i++)if(!Finite(x[i]))return false;
        return true;
    }
    static void RepairHessian(double[,] h,double[] g,bool diagonal)
    {
        int n=g.Length;double scale=1,minDiagonal=double.PositiveInfinity,maxDiagonal=0;
        for(int i=0;i<n;i++) {minDiagonal=Math.Min(minDiagonal,Math.Abs(h[i,i]));maxDiagonal=Math.Max(maxDiagonal,Math.Abs(h[i,i]));}
        if(maxDiagonal>1e12*Math.Max(1e-12,minDiagonal))diagonal=true;
        for(int i=0;i<n;i++)for(int j=0;j<n;j++) {
            if(!Finite(h[i,j]))diagonal=true;
            else scale=Math.Max(scale,Math.Abs(h[i,j]));
        }
        if(diagonal) {
            Array.Clear(h,0,h.Length);
            for(int i=0;i<n;i++)h[i,i]=2.5*Math.Max(1,Math.Abs(g[i]));
            return;
        }
        // Symmetric strict diagonal dominance gives a bounded SPD repair.
        for(int i=0;i<n;i++)for(int j=0;j<i;j++)h[i,j]=h[j,i]=.5*h[i,j]+.5*h[j,i];
        double shift=0,margin=1e-8*scale;
        for(int i=0;i<n;i++) {
            double radius=0;for(int j=0;j<n;j++)if(j!=i)radius+=Math.Abs(h[i,j]);
            shift=Math.Max(shift,radius-h[i,i]+margin);
        }
        for(int i=0;i<n;i++)h[i,i]+=shift;
    }
    static bool Finite(double x)=>!double.IsNaN(x)&&!double.IsInfinity(x);
    public static bool ValidCost(double cost)=>Finite(cost)&&cost<1e19;
    public static bool ValidEvaluation(double cost,double[,] gradient) {
        if(!ValidCost(cost))return false;
        if(gradient!=null)foreach(double v in gradient)if(!Finite(v))return false;
        return true;
    }
    static bool Bounded(double[,] sequence) {
        foreach(double v in sequence)if(!Finite(v)||v < -1.000000001||v > 1.000000001)return false;
        return true;
    }
    public static double VerifyActiveSet()
    {
        var h=new double[,]{{4,1,0},{1,3,.5},{0,.5,2}};double worst=0;
        foreach(int sign in new[]{-1,1})foreach(bool corner in new[]{false,true}) {
            var target=new[]{sign*1.8,sign*(-.2),sign*.4};var expected=new[]{(double)sign,sign*(-.2+1.6/5.75),sign*(.4-.4/5.75)};
            var u=corner?new double[,]{{-sign,sign,-sign}}:new double[1,3];
            Func<double[,],double[,],double> objective=(x,gradient)=>{
                double cost=0;for(int i=0;i<3;i++){double value=0;for(int j=0;j<3;j++)value+=h[i,j]*(x[0,j]-target[j]);cost+=.5*(x[0,i]-target[i])*value;if(gradient!=null)gradient[0,i]=value;}return cost;};
            var result=Solve(u,objective,32,h,true);
            for(int i=0;i<3;i++)worst=Math.Max(worst,Math.Abs(u[0,i]-expected[i]));
            worst=Math.Max(worst,result.ProjectedGradient);
            worst=Math.Max(worst,result.QpResidual);
        }
        if(worst>1e-6)throw new InvalidOperationException("Active-set analytic QP check failed "+worst);
        return worst;
    }
    static double[] Flatten(double[,] matrix) {
        var result=new double[matrix.Length];int cols=matrix.GetLength(1);
        for(int i=0;i<result.Length;i++)result[i]=matrix[i/cols,i%cols];return result;
    }
    static double Stationarity(double[] x,double[] g) {
        double error=0;
        for(int i=0;i<x.Length;i++)error=Math.Max(error,Math.Abs(x[i]-Math.Max(-1,Math.Min(1,x[i]-g[i]))));
        return error;
    }
}
