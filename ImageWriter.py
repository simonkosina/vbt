import cv2


class ImageWriter(object):
    def __init__(
            self,
            image,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            textColor=(255, 255, 255),
            backgroundColor=(115, 3, 252),
            thickness=2,
            lineType=cv2.LINE_AA,
            horizontalPadding=10,
            verticalPadding=10
            ):
        self.image = image
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.textColor = textColor
        self.backgroundColor = backgroundColor
        self.thickness = thickness
        self.lineType = lineType
        self.horizontalPadding = horizontalPadding
        self.verticalPadding = verticalPadding

        self.n_lines = 0

    def putText(self, text):
        (text_width, text_height), _ = cv2.getTextSize(
            text=text,
            fontFace=self.fontFace,
            fontScale=self.fontScale,
            thickness=self.thickness
        )

        y_start = self.n_lines*(text_height+self.horizontalPadding)
        y_end = (self.n_lines+1)*(text_height+self.horizontalPadding)

        cv2.rectangle(
            img=self.image,
            pt1=(0, y_start),
            pt2=(self.verticalPadding + text_width, y_end),
            color=self.backgroundColor,
            thickness=-1
        )

        cv2.putText(
            img=self.image,
            text=text,
            org=(int(self.verticalPadding/2),
                 y_end + self.fontScale - 1 - int(self.horizontalPadding/2)),
            fontFace=self.fontFace,
            fontScale=self.fontScale,
            color=self.textColor,
            thickness=self.thickness,
            lineType=self.lineType
        )

        self.n_lines += 1
