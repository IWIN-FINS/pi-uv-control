"""Exact historical objective replay and an optional, separate SciPy solver demo.

The demo is NOT the original native SQP and has no historical timing/closed-loop
equivalence claim. The objective is tested against a nonzero-bias native record.
"""
import numpy as np
from .geometry import rate_matrix
from .scales import PS, PC, VS


def objective_batch(model, record, sequences):
    x = np.asarray(sequences, dtype=np.float64).reshape(-1, 8, 8)
    count = len(x)
    z = np.repeat(np.asarray(record['z'])[None], count, axis=0)
    force = np.asarray(record['forceCenter']) + x * np.asarray(record['forceHalf'])
    body = force @ np.asarray(record['allocationBasis']).reshape(6, 8).T
    inputs = body / np.asarray(record['wrenchLimit'])
    refs = np.asarray(record['referencePose']).reshape(8, 6)
    vrefs = np.asarray(record['referenceVelocity']).reshape(8, 6)
    weights = np.array([1.]*7 + [3.])
    if np.any(record.get('de92Matrix',np.zeros(60))) or np.any(record.get('de83PoseBias',np.zeros(6))):
        raise ValueError('Mainline objective supports velocity bias only, not C or pose updates')
    cost = np.zeros(count)
    for j in range(8):
        z = model.step(z, inputs[:, j], record['de83VelocityBias'])
        p = z[:,1:7]*PS+PC
        if not np.isfinite(z).all() or np.max(abs(z[:,1:])) > 100 or np.max(abs(p[:,3])) >= 1.3:
            raise ValueError('Prediction outside historical chart/feasibility domain')
        rates = np.einsum('nij,nj->ni',rate_matrix(p),z[:,7:13]*VS)
        cost += weights[j] * (np.sum(np.asarray(record['positionWeight'])*(p-refs[j])**2,axis=1)
                             + np.sum(np.asarray(record['velocityWeight'])*(rates-vrefs[j])**2,axis=1))
        cost += np.sum(np.asarray(record['effortWeight'])*body[:,j]**2,axis=1)+.0005*np.sum(force[:,j]**2,axis=1)
    return cost


def central_gradient(model, record, x, epsilon=1e-5):
    x = np.asarray(x, dtype=np.float64).reshape(64)
    probes = np.r_[x[None]+epsilon*np.eye(64),x[None]-epsilon*np.eye(64)]
    values = objective_batch(model, record, probes)
    return (values[:64]-values[64:])/(2*epsilon)


def solve_reference(model, record, max_iterations=128):
    """Inspection-only solve of a captured objective; never sends commands."""
    from scipy.optimize import minimize
    return minimize(lambda x:float(objective_batch(model,record,x)[0]),
                    np.array(record['start']),
                    jac=lambda x:central_gradient(model,record,x),
                    method='SLSQP',bounds=[(-1.,1.)]*64,
                    options={'maxiter':max_iterations,'ftol':1e-5})
