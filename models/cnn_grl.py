from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.models import Model

from .base_model import BaseModel

class CNNGRL(BaseModel):
    def _build(self, num_classes, num_domains=2):
        inputs, features = self._build_feature_extractor()
        label_predictions = self._build_label_predictor(features, num_classes)
        domain_predictions = self._build_domain_classifier(features, num_domains)
        self.model = Model(inputs=inputs, outputs=[label_predictions, domain_predictions])

    def _build_feature_extractor(self):
        inputs = Input(shape=(32, 32, 3))
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        features = Flatten()(x)
        return inputs, features

    def _build_label_predictor(self, features, num_classes):
        x = Dense(512, activation='relu')(features)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax', name='label_predictor')(x)
        return outputs

    def _build_domain_classifier(self, features, num_domains):
        #TODO: add GRL
        x = Dense(512, activation='relu')(features)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_domains, activation='softmax', name='domain_classifier')(x)
        return outputs




