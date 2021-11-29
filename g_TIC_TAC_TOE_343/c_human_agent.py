class Human_Agent:
    def __init__(self, name, env):
        self.name = name
        self.env = env

    def get_action(self, current_state):
        available_actions = current_state.get_available_actions()
        valid_action_id = False
        action_id = None

        while not valid_action_id:
            action_id = int(input(
                "12개 셀 중 하나를 선택하세요 "
                "([8,9,10,11,  4,5,6,7,  0,1,2,3] 숫자 중 하나를 선택하고 엔터를 누르세요)"
            ))
            if action_id > 11 or action_id < 0:
                print("[입력 오류: {0}] 0부터 11사이의 숫자 값을 입력하세요.".format(
                    action_id
                ))
                continue

            if action_id not in available_actions:
                print("[입력 오류: {0}] 유효한 셀을 선택하세요.".format(action_id))
            else:
                valid_action_id = True

        return action_id