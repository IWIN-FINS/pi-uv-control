using System;
using System.Collections.Generic;

// Controlled, finite-dimensional EDMD operator for the complete six-axis state.
// z = [1, normalized pose(6), normalized velocity(6), velocity*abs(velocity)(6)].
// z[k+1] = A z[k] + B normalizedWrench[k]. The constant and kinematic rows
// are constrained; the velocity and nonlinear-observable rows are fitted with
// a ridge prior centered on the calibrated Fossen discrete-time model.
public sealed class FinsRovLiftedOperator
{
    public const int Axes = 6;
    public const int Lifted = 19;
    public const int Regressors = Lifted + Axes;
    public const float Dt = 0.1f;
    public static readonly float[] PoseScale = { 0.6f, 0.2f, 0.35f, 0.2f, 0.4f, 0.2f };
    public static readonly float[] PoseCenter = { 0f, -1.2f, 0f, 0f, 0f, 0f };
    public static readonly float[] VelocityScale = { 0.15f, 0.08f, 0.15f, 0.25f, 0.4f, 0.25f };
    public static readonly float[] WrenchScale = { 15f, 12f, 15f, 2f, 2f, 2f };

    public sealed class Transition
    {
        public readonly float[] Pose, Velocity, Wrench, NextPose, NextVelocity;
        public readonly int Segment;
        public Transition(float[] pose, float[] velocity, float[] wrench,
            float[] nextPose, float[] nextVelocity, int segment)
        {
            Pose = (float[])pose.Clone(); Velocity = (float[])velocity.Clone();
            Wrench = (float[])wrench.Clone(); NextPose = (float[])nextPose.Clone();
            NextVelocity = (float[])nextVelocity.Clone(); Segment = segment;
        }
    }

    public readonly double[,] A = new double[Lifted, Lifted];
    public readonly double[,] B = new double[Lifted, Axes];

    public static double[] Lift(float[] pose, float[] velocity)
    {
        var z = new double[Lifted];
        z[0] = 1d;
        for (int a = 0; a < Axes; a++)
        {
            z[1 + a] = (pose[a] - PoseCenter[a]) / PoseScale[a];
            z[7 + a] = velocity[a] / VelocityScale[a];
            z[13 + a] = z[7 + a] * Math.Abs(z[7 + a]);
        }
        return z;
    }

    public static double[] Input(float[] wrench)
    {
        var result = new double[Axes];
        for (int a = 0; a < Axes; a++) result[a] = wrench[a] / WrenchScale[a];
        return result;
    }

    public static double DecodePose(double[] z, int axis) =>
        PoseCenter[axis] + PoseScale[axis] * z[1 + axis];
    public static double DecodeVelocity(double[] z, int axis) =>
        VelocityScale[axis] * z[7 + axis];

    public void BuildNominalPrior(float[] mass, float[] linearDrag,
        float[] quadraticDrag, float[] spring, float[] bias)
    {
        A[0, 0] = 1d;
        for (int axis = 0; axis < Axes; axis++)
        {
            int p = 1 + axis, v = 7 + axis, q = 13 + axis;
            A[v, 0] = Dt * (bias[axis] - spring[axis] * PoseCenter[axis]) /
                (mass[axis] * VelocityScale[axis]);
            A[v, p] = -Dt * spring[axis] * PoseScale[axis] /
                (mass[axis] * VelocityScale[axis]);
            A[v, v] = 1d - Dt * linearDrag[axis] / mass[axis];
            A[v, q] = -Dt * quadraticDrag[axis] * VelocityScale[axis] / mass[axis];
            B[v, axis] = Dt * WrenchScale[axis] / (mass[axis] * VelocityScale[axis]);
            A[q, q] = 1d;
        }
        SetKinematicRows();
    }

    void SetKinematicRows()
    {
        for (int axis = 0; axis < Axes; axis++)
        {
            int p = 1 + axis, v = 7 + axis;
            for (int j = 0; j < Lifted; j++)
                A[p, j] = (j == p ? 1d : 0d) +
                    Dt * VelocityScale[axis] / PoseScale[axis] * A[v, j];
            for (int j = 0; j < Axes; j++)
                B[p, j] = Dt * VelocityScale[axis] / PoseScale[axis] * B[v, j];
        }
    }

    public void Fit(IReadOnlyList<Transition> samples, int trainCount)
    {
        if (trainCount < 300 || samples.Count <= trainCount + 100)
            throw new InvalidOperationException("EDMDc requires disjoint, sufficiently long training and validation segments");
        var gram = new double[Regressors, Regressors];
        var feature = new double[trainCount][];
        var target = new double[trainCount][];
        for (int k = 0; k < trainCount; k++)
        {
            feature[k] = new double[Regressors];
            double[] lifted = Lift(samples[k].Pose, samples[k].Velocity);
            double[] input = Input(samples[k].Wrench);
            Array.Copy(lifted, feature[k], Lifted);
            Array.Copy(input, 0, feature[k], Lifted, Axes);
            target[k] = Lift(samples[k].NextPose, samples[k].NextVelocity);
            for (int i = 0; i < Regressors; i++)
                for (int j = 0; j < Regressors; j++)
                    gram[i, j] += feature[k][i] * feature[k][j];
        }
        for (int output = 7; output < Lifted; output++)
        {
            var rhs = new double[Regressors];
            for (int k = 0; k < trainCount; k++)
            {
                double priorPrediction = 0d;
                for (int j = 0; j < Lifted; j++)
                    priorPrediction += A[output, j] * feature[k][j];
                for (int j = 0; j < Axes; j++)
                    priorPrediction += B[output, j] * feature[k][Lifted + j];
                double residual = target[k][output] - priorPrediction;
                for (int j = 0; j < Regressors; j++) rhs[j] += feature[k][j] * residual;
            }
            double lambda = output < 13 ? 0.30d : 1.0d;
            var regularized = (double[,])gram.Clone();
            for (int j = 0; j < Regressors; j++) regularized[j, j] += lambda;
            double[] delta = Solve(regularized, rhs);
            for (int j = 0; j < Lifted; j++) A[output, j] += delta[j];
            for (int j = 0; j < Axes; j++) B[output, j] += delta[Lifted + j];
        }
        SetKinematicRows();
    }

    public double[] Step(double[] z, double[] normalizedWrench)
    {
        var next = new double[Lifted];
        for (int i = 0; i < Lifted; i++)
        {
            double value = 0d;
            for (int j = 0; j < Lifted; j++) value += A[i, j] * z[j];
            for (int j = 0; j < Axes; j++) value += B[i, j] * normalizedWrench[j];
            next[i] = value;
        }
        return next;
    }

    // Diagnostic interventions after fitting, without changing the training set.
    public void RestrictRotationCoupling()
    {
        for (int axis = 3; axis < Axes; axis++)
            foreach (int row in new[] { 7 + axis, 13 + axis })
            {
                for (int col = 1; col < Lifted; col++)
                    if (col != 1 + axis && col != 7 + axis && col != 13 + axis)
                        A[row, col] = 0d;
                for (int col = 0; col < Axes; col++)
                    if (col != axis) B[row, col] = 0d;
            }
        SetKinematicRows();
    }

    public void RemovePitchAccelerationBias()
    {
        A[12, 0] = 0d;
        SetKinematicRows();
    }

    public void RemoveRotationHorizontalPosition()
    {
        for (int axis = 3; axis < Axes; axis++)
            foreach (int row in new[] { 7 + axis, 13 + axis })
                A[row,1] = A[row,3] = 0d;
        SetKinematicRows();
    }

    public void RemoveRotationCrossInput()
    {
        for (int axis = 3; axis < Axes; axis++)
            foreach (int row in new[] { 7 + axis, 13 + axis })
                for (int col = 0; col < Axes; col++)
                    if (col != axis) B[row,col] = 0d;
        SetKinematicRows();
    }

    public static double[] Solve(double[,] matrix, double[] rhs)
    {
        int n = rhs.Length;
        var a = (double[,])matrix.Clone();
        var b = (double[])rhs.Clone();
        for (int column = 0; column < n; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < n; row++)
                if (Math.Abs(a[row, column]) > Math.Abs(a[pivot, column])) pivot = row;
            if (Math.Abs(a[pivot, column]) < 1e-10)
                throw new InvalidOperationException("Singular EDMDc regression matrix");
            for (int j = column; j < n; j++)
            {
                double swap = a[column, j]; a[column, j] = a[pivot, j]; a[pivot, j] = swap;
            }
            double rightSwap = b[column]; b[column] = b[pivot]; b[pivot] = rightSwap;
            double inverse = 1d / a[column, column];
            for (int j = column; j < n; j++) a[column, j] *= inverse;
            b[column] *= inverse;
            for (int row = 0; row < n; row++)
            {
                if (row == column) continue;
                double factor = a[row, column];
                for (int j = column; j < n; j++) a[row, j] -= factor * a[column, j];
                b[row] -= factor * b[column];
            }
        }
        return b;
    }
}
