#
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from datetime import datetime

import pytest

from otx.api.serialization.datetime_mapper import DatetimeMapper
from otx.api.tests.constants.otx.api_components import OtxApiComponent
from otx.api.tests.constants.requirements import Requirements
from otx.api.utils.time_utils import now


@pytest.mark.components(OtxApiComponent.OTX_API)
class TestDatetimeMapper:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_serialization_deserialization(self):
        """
        This test serializes datetime, deserializes serialized datetime and compares with original one.
        """

        original_time = now()
        serialized_time = DatetimeMapper.forward(original_time)
        assert serialized_time == original_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        deserialized_time = DatetimeMapper.backward(serialized_time)
        assert original_time == deserialized_time

        deserialized_time = DatetimeMapper.backward(None)
        assert isinstance(deserialized_time, datetime)
