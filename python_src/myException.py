# 動画ファイルが開けなかった場合の例外
class VideoOpenError(Exception):
    pass

# ラベルマップの設定が誤っている場合の例外
class LabelMapSettingError(Exception):
    pass