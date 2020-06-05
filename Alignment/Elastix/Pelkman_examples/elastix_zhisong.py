import SimpleITK as sitk
from typing import Union, Tuple


def reg_trans_affine_bspline32(c1: sitk.Image, c2: sitk.Image) -> Tuple[sitk.Image, sitk.ParameterMap]:
    elstx = sitk.ElastixImageFilter()
    elstx.SetFixedImage(c1)
    elstx.SetMovingImage(c2)
    elstx.SetParameterMap(sitk.ReadParameterFile('reg_params/trans.txt'))
    elstx.AddParameterMap(sitk.ReadParameterFile('reg_params/affine.txt'))
    elstx.AddParameterMap(sitk.ReadParameterFile('reg_params/bspline_32grid.txt'))
    out = elstx.Execute()
    trans = elstx.GetTransformParameterMap()
    return out, trans


def apply_transform(img: sitk.Image, transform: Union[sitk.ParameterMap, sitk.VectorOfParameterMap]) -> sitk.Image:
    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(img)
    transformix.SetTransformParameterMap(transform)
    return transformix.Execute()