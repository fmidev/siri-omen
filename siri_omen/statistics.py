"""
Statistical metrics.
"""
import numpy


def union_mask(reference, predicted):
    def _get_mask(s):
        if numpy.ma.is_masked(s):
            return ~numpy.isfinite(s.filled(numpy.nan))
        return ~numpy.isfinite(s)
    return _get_mask(reference) + _get_mask(predicted)


def compute_error(reference, predicted):
    return predicted - reference


def compute_bias(reference, predicted):
    err = compute_error(reference, predicted)
    return numpy.mean(err)


def compute_rmse(reference, predicted):
    err = compute_error(reference, predicted)
    return numpy.sqrt(numpy.mean(err**2))


def compute_crmse(reference, predicted):
    err = compute_error(reference - numpy.mean(reference),
                        predicted - numpy.mean(predicted))
    return numpy.sqrt(numpy.mean(err**2))


def compute_stddev(reference):
    return numpy.std(reference, ddof=1)


def compute_corrcoef(reference, predicted):
    return numpy.corrcoef(predicted, reference)[0, 1]


def compute_stat(kind, reference, predicted):
    op = {
        'bias': compute_bias,
        'rmse': compute_rmse,
        'crmse': compute_crmse,
        'stddev': compute_stddev,
        'corrcoef': compute_corrcoef,
    }
    args = (reference, predicted)
    if kind == 'stddev':
        args = (predicted, )
    return op[kind](*args)


def _compute_all_stats(reference, predicted):
    op_list = ['bias', 'rmse', 'crmse', 'stddev', 'corrcoef']
    return dict([(o, compute_stat(o, reference, predicted)) for o in op_list])


def compute_statistics(reference, predicted):

    r = numpy.copy(reference)
    p = numpy.copy(predicted)

    # keep only values that are valid in both sets
    mask = union_mask(r, p)
    r = r[~mask]
    p = p[~mask]

    ref_stats = _compute_all_stats(r, r)
    pre_stats = _compute_all_stats(r, p)

    return ref_stats, pre_stats


def normalize_statistics(ref_stats, pre_stats, add_crmse_sign=False):

    norm_metrics = ['bias', 'rmse', 'crmse', 'stddev']

    def _make_norm_dict(d, scalar):
        new = dict(d)
        for key in norm_metrics:
            new[key] *= scalar
        return new

    scalar = 1.0 / ref_stats['stddev']
    nref_stats = _make_norm_dict(ref_stats, scalar)
    npre_stats = _make_norm_dict(pre_stats, scalar)

    if add_crmse_sign:
        crmse_sign = numpy.sign(pre_stats['stddev'] - ref_stats['stddev'])
        npre_stats['crmse'] *= crmse_sign

    return nref_stats, npre_stats
