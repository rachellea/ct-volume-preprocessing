#test_combine_slices.py
#modified from https://github.com/innolitics/dicom-numpy

import numpy as np

from combine_slices import combine_slices_func, _validate_slices_form_uniform_grid, _merge_slice_pixel_arrays, DicomImportException

# direction cosines
x_cos = (1, 0, 0)
y_cos = (0, 1, 0)
z_cos = (0, 0, 1)
negative_x_cos = (-1, 0, 0)
negative_y_cos = (0, -1, 0)
negative_z_cos = (0, 0, -1)

arbitrary_shape = (10, 11)

class MockSlice:
    '''
    A minimal DICOM dataset representing a dataslice at a particular
    slice location.  The `slice_position` is the coordinate value along the
    remaining unused axis (i.e. the axis perpendicular to the direction
    cosines).
    '''

    def __init__(self, pixel_array, slice_position, row_cosine=None, column_cosine=None, **kwargs):
        if row_cosine is None:
            row_cosine = x_cos

        if column_cosine is None:
            column_cosine = y_cos

        na, nb = pixel_array.shape

        self.pixel_array = pixel_array

        self.SeriesInstanceUID = 'arbitrary uid'
        self.SOPClassUID = 'arbitrary sopclass uid'
        self.PixelSpacing = [1.0, 1.0]
        self.Rows = na
        self.Columns = nb
        self.Modality = 'MR'

        # assume that the images are centered on the remaining unused axis
        a_component = [-na/2.0*c for c in row_cosine]
        b_component = [-nb/2.0*c for c in column_cosine]
        c_component = [(slice_position if c == 0 and cc == 0 else 0) for c, cc in zip(row_cosine, column_cosine)]
        patient_position = [a + b + c for a, b, c in zip(a_component, b_component, c_component)]

        self.ImagePositionPatient = patient_position

        self.ImageOrientationPatient = list(row_cosine) + list(column_cosine)

        for k, v in kwargs.items():
            setattr(self, k, v)

def axial_slices():
    return [MockSlice(randi(*arbitrary_shape), 0),
        MockSlice(randi(*arbitrary_shape), 1),
        MockSlice(randi(*arbitrary_shape), 2),
        MockSlice(randi(*arbitrary_shape), 3),]

def randi(*shape):
    return np.random.randint(1000, size=shape, dtype='uint16')

class TestCombineSlices:
    def __init__(self):
        self.test_simple_axial_set(axial_slices())
    
    def test_simple_axial_set(self, axial_slices):
        combined, _, _ = combine_slices_func(axial_slices[0:2])

        manually_combined = np.dstack((axial_slices[1].pixel_array.T, axial_slices[0].pixel_array.T))
        assert np.array_equal(combined, manually_combined)
        print('Passed test_simple_axial_set()')


class TestMergeSlicePixelArrays:
    def __init__(self):
        self.test_casting_if_only_rescale_slope()
        self.test_casting_rescale_slope_and_intercept()
        self.test_robust_to_ordering(axial_slices())
        self.test_rescales_if_forced_true()
        self.test_no_rescale_if_forced_false()
    
    def test_casting_if_only_rescale_slope(self):
        '''
        If the `RescaleSlope` DICOM attribute is present, the
        `RescaleIntercept` attribute should also be present, however, we handle
        this case anyway.
        '''
        slices = [
            MockSlice(np.ones((10, 20), dtype=np.uint8), 0, RescaleSlope=2),
            MockSlice(np.ones((10, 20), dtype=np.uint8), 1, RescaleSlope=2),
        ]

        voxels, _ = _merge_slice_pixel_arrays(slices)
        assert voxels.dtype == np.dtype('float32')
        assert voxels[0, 0, 0] == 2.0
        print('Passed test_casting_if_only_rescale_slope()')

    def test_casting_rescale_slope_and_intercept(self):
        '''
        Some DICOM modules contain the `RescaleSlope` and `RescaleIntercept` DICOM attributes.
        '''
        slices = [
            MockSlice(np.ones((10, 20), dtype=np.uint8), 0, RescaleSlope=2, RescaleIntercept=3),
            MockSlice(np.ones((10, 20), dtype=np.uint8), 1, RescaleSlope=2, RescaleIntercept=3),
        ]

        voxels, _ = _merge_slice_pixel_arrays(slices)
        assert voxels.dtype == np.dtype('float32')
        assert voxels[0, 0, 0] == 5.0
        print('Passed test_casting_rescale_slope_and_intercept()')

    def test_robust_to_ordering(self, axial_slices):
        '''
        The DICOM slices should be able to be passed in in any order, and they
        should be recombined appropriately.
        '''
        a, _ = _merge_slice_pixel_arrays([axial_slices[0], axial_slices[1], axial_slices[2]])
        b, _ = _merge_slice_pixel_arrays([axial_slices[1], axial_slices[0], axial_slices[2]])
        assert np.array_equal(a,b)

        c, _ = _merge_slice_pixel_arrays([axial_slices[0], axial_slices[1], axial_slices[2]])
        d, _ = _merge_slice_pixel_arrays([axial_slices[2], axial_slices[0], axial_slices[1]])
        assert np.array_equal(c,d)
        print('Passed test_robust_to_ordering()')

    def test_rescales_if_forced_true(self):
        slice_datasets = [MockSlice(np.ones((10, 20), dtype=np.uint8), 0)]
        voxels, _ = _merge_slice_pixel_arrays(slice_datasets, rescale=True)
        assert voxels.dtype == np.float32
        print('Passed test_rescales_if_forced_true')

    def test_no_rescale_if_forced_false(self):
        slice_datasets = [MockSlice(np.ones((10, 20), dtype=np.uint8), 0, RescaleSlope=2, RescaleIntercept=3)]
        voxels, _ = _merge_slice_pixel_arrays(slice_datasets, rescale=False)
        assert voxels.dtype == np.uint8
        print('Passed test_no_rescale_if_forced_false()')


class TestValidateSlicesFormUniformGrid:
    def __init__(self):
        #self.test_missing_middle_slice(axial_slices())
        self.test_insignificant_difference_in_direction_cosines(axial_slices())
        self.test_significant_difference_in_direction_cosines(axial_slices())
        self.test_slices_from_different_series(axial_slices())
    
    # def test_missing_middle_slice(self, axial_slices):
    #     '''
    #     All slices must be present.  Slice position is determined using the
    #     ImagePositionPatient (0020,0032) tag.
    #     '''
    #     #Pass the test only if this raises DicomImportException
    #     try:
    #         _validate_slices_form_uniform_grid([axial_slices[0], axial_slices[2], axial_slices[3]])
    #     except DicomImportException:
    #         print('Passed test_missing_middle_slice()')
    #         return
    #     raise Exception

    def test_insignificant_difference_in_direction_cosines(self, axial_slices):
        '''
        We have seen DICOM series in the field where slices have lightly
        different direction cosines.
        '''
        axial_slices[0].ImageOrientationPatient[0] += 1e-6
        _validate_slices_form_uniform_grid(axial_slices)
        print('Passed test_insignificant_difference_in_direction_cosines()')

    def test_significant_difference_in_direction_cosines(self, axial_slices):
        #Pass the test only if this raises DicomImportException
        axial_slices[0].ImageOrientationPatient[0] += 1e-4
        try: 
            _validate_slices_form_uniform_grid(axial_slices)
        except DicomImportException:
            print('Passed test_significant_difference_in_direction_cosines()')
            return
        raise Exception

    def test_slices_from_different_series(self, axial_slices):
        '''
        As a sanity check, slices that don't come from the same DICOM series should
        be rejected.
        '''
        #Pass the test only if this raises DicomImportException
        axial_slices[2].SeriesInstanceUID += 'Ooops'
        try:
            _validate_slices_form_uniform_grid(axial_slices)
        except DicomImportException:
            print('Passed test_slices_from_different_series()')
            return
        raise Exception

if __name__ == '__main__':
    TestCombineSlices()
    TestMergeSlicePixelArrays()
    TestValidateSlicesFormUniformGrid()
    