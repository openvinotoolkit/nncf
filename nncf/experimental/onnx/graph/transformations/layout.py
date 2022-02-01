from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType


class ONNXTransformationLayout(TransformationLayout):
    def register(self, transformation: TransformationCommand) -> None:
        if transformation.type == TransformationType.INSERT:
            self.transformations.append(transformation)
