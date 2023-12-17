train_labels = [
    'ApplyEyeMakeup', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching',
    'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking',
    'Billiards', 'BlowDryHair', 'BodyWeightSquats', 'Bowling',
    'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
    'CricketBowling', 'Drumming', 'Fencing', 'FieldHockeyPenalty',
    'FrisbeeCatch', 'FrontCrawl', 'Haircut', 'Hammering', 'HeadMassage',
    'HulaHoop', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'Kayaking',
    'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing',
    'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing',
    'PlayingCello', 'PlayingDhol', 'PlayingFlute', 'PlayingPiano',
    'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PullUps',
    'PushUps', 'Rafting', 'RopeClimbing', 'Rowing', 'ShavingBeard', 'Skijet',
    'SoccerJuggling', 'SoccerPenalty', 'SumoWrestling', 'Swing',
    'TableTennisShot', 'TaiChi', 'ThrowDiscus', 'TrampolineJumping', 'Typing',
    'UnevenBars', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo'
]

val_labels = [
    'ApplyLipstick', 'CricketShot', 'HammerThrow', 'HandstandPushups',
    'HighJump', 'HorseRiding', 'PlayingDaf', 'PlayingGuitar', 'Shotput',
    'SkateBoarding'
]

test_labels = [
    'BlowingCandles', 'CleanAndJerk', 'CliffDiving', 'CuttingInKitchen',
    'Diving', 'FloorGymnastics', 'GolfSwing', 'HandstandWalking', 'HorseRace',
    'IceDancing', 'JumpRope', 'PommelHorse', 'Punch', 'RockClimbingIndoor',
    'SalsaSpin', 'Skiing', 'SkyDiving', 'StillRings', 'Surfing', 'TennisSwing',
    'VolleyballSpiking'
]
'''
read all lines in ucf101_train_split_1_videos.txt
split by '/' and take the first element
if it is in train_labels, write that line to train.txt
if it is in val_labels, write that line to val.txt
if it is in test_labels, write that line to test.txt
'''
with open('ucf101_train_split_1_videos.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        label = line.split('/')[0]
        if label in train_labels:
            with open('train.txt', 'a') as f1:
                f1.write(line + '\n')
        elif label in val_labels:
            with open('val.txt', 'a') as f2:
                f2.write(line + '\n')
        elif label in test_labels:
            with open('test.txt', 'a') as f3:
                f3.write(line + '\n')
        else:
            print('Error: label not found: ' + label)
with open('ucf101_val_split_1_videos.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        label = line.split('/')[0]
        if label in train_labels:
            with open('train.txt', 'a') as f1:
                f1.write(line + '\n')
        elif label in val_labels:
            with open('val.txt', 'a') as f2:
                f2.write(line + '\n')
        elif label in test_labels:
            with open('test.txt', 'a') as f3:
                f3.write(line + '\n')
        else:
            print('Error: label not found: ' + label)
