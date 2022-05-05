from albumentations import Compose
import random

class TemporalCompose(Compose):
  def __init__(self, *args, **kwargs):
    super(TemporalCompose, self).__init__(*args, **kwargs)
    # First set the probability of all transforms to zero
    # since we will be re-writing the logic for the probability
    self.probabilities = ([t.p for t in self.transforms])
    for transform in self.transforms:
      transform.p = 0
    self.preComputeTransformApplication()

  def preComputeTransformApplication(self):
    self.applyT = []
    for idx, t in enumerate(self.transforms):
      self.applyT.append((random.random() < self.probabilities[idx]) or t.always_apply)
      # Since some transforms are left as is, they will not have the preCompute function
      try:
        t.preCompute()
      except AttributeError:
        continue

  def __call__(self, *args, force_apply: bool = False, **data):
      if args:
          raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
      if self.is_check_args:
          self._check_args(**data)
      assert isinstance(force_apply, (bool, int)), "force_apply must have bool or int type"
      need_to_run = force_apply or random.random() < self.p
      for p in self.processors.values():
          p.ensure_data_valid(data)
      transforms = self.transforms if need_to_run else get_always_apply(self.transforms)

      check_each_transform = any(
          getattr(item.params, "check_each_transform", False) for item in self.processors.values()
      )

      for p in self.processors.values():
          p.preprocess(data)

      for idx, t in enumerate(transforms):
          data = t(force_apply=self.applyT[idx], **data)

          if check_each_transform:
              data = self._check_data_post_transform(data)
      # data = self._make_targets_contiguous(data)  # ensure output targets are contiguous

      for p in self.processors.values():
          p.postprocess(data)

      return data
