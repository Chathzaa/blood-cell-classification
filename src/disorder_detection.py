"""Compatibility endpoint for the retired, unvalidated diagnostic rules."""


def detect_disorders(cell_counts, feature_list, **kwargs):
    return {
        'status':'not_assessed',
        'reason':('The available data do not validate diagnosis of leukemia, anemia or sickle cell '
                  'disease from these image counts and stain measurements.'),
    }
