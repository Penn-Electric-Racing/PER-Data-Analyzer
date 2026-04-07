import cmd
import time

from colorama import Fore, Style, init

from perda.live import LiveAnalyzer, ValueType

init(autoreset=True)

live = LiveAnalyzer.local()

PARAMS = {
    "kp": "ludwig.tunable.tractionControl.kp",
    "ki": "ludwig.tunable.tractionControl.ki",
    "kd": "ludwig.tunable.tractionControl.kd",
    "kf": "ludwig.tunable.tractionControl.kf",
}


class TractionShell(cmd.Cmd):
    intro = (
        Fore.CYAN + "Traction Control Tuning Shell. Type help or ?" + Style.RESET_ALL
    )
    prompt = Fore.YELLOW + "tc> " + Style.RESET_ALL

    def get_val(self, key):
        return live.get(PARAMS[key], ValueType.FLOAT)

    def set_val(self, key, val):
        live.set(PARAMS[key], val, ValueType.FLOAT)

    def do_show(self, arg):
        """show current traction control parameters"""
        for k in PARAMS:
            try:
                v = self.get_val(k)
                print(
                    f"{Fore.BLUE}{k:>2}{Style.RESET_ALL} = "
                    f"{Fore.GREEN}{v:.4f}{Style.RESET_ALL}"
                )
            except Exception as e:
                print(Fore.RED + f"{k}: error ({e})")

    def do_set(self, arg):
        """set <param> <value>"""
        try:
            p, v = arg.split()
            v = float(v)

            if p not in PARAMS:
                print(Fore.RED + "unknown param")
                return

            self.set_val(p, v)

            print(
                f"{Fore.BLUE}{p}{Style.RESET_ALL} -> "
                f"{Fore.GREEN}{v}{Style.RESET_ALL}"
            )

        except ValueError:
            print(Fore.RED + "usage: set <kp|ki|kd|kf> <value>")

    def do_inc(self, arg):
        """inc <param> <amount>"""
        try:
            p, step = arg.split()
            step = float(step)

            if p not in PARAMS:
                print(Fore.RED + "unknown param")
                return

            val = self.get_val(p)
            new = val + step
            self.set_val(p, new)

            print(
                f"{Fore.BLUE}{p}{Style.RESET_ALL}: "
                f"{Fore.WHITE}{val:.4f}{Style.RESET_ALL} -> "
                f"{Fore.GREEN}{new:.4f}{Style.RESET_ALL}"
            )

        except Exception:
            print(Fore.RED + "usage: inc <param> <amount>")

    def do_dec(self, arg):
        """dec <param> <amount>"""
        try:
            p, step = arg.split()
            step = float(step)

            if p not in PARAMS:
                print(Fore.RED + "unknown param")
                return

            val = self.get_val(p)
            new = val - step
            self.set_val(p, new)

            print(
                f"{Fore.BLUE}{p}{Style.RESET_ALL}: "
                f"{Fore.WHITE}{val:.4f}{Style.RESET_ALL} -> "
                f"{Fore.RED}{new:.4f}{Style.RESET_ALL}"
            )

        except Exception:
            print(Fore.RED + "usage: dec <param> <amount>")

    def do_watch(self, arg):
        """watch parameters continuously"""
        try:
            while True:
                vals = [self.get_val(k) for k in PARAMS]

                print(
                    f"{Fore.BLUE}kp{Style.RESET_ALL}={Fore.GREEN}{vals[0]:.3f}  "
                    f"{Fore.BLUE}ki{Style.RESET_ALL}={Fore.GREEN}{vals[1]:.3f}  "
                    f"{Fore.BLUE}kd{Style.RESET_ALL}={Fore.GREEN}{vals[2]:.3f}  "
                    f"{Fore.BLUE}kf{Style.RESET_ALL}={Fore.GREEN}{vals[3]:.3f}",
                    end="\r",
                )

                time.sleep(0.2)

        except KeyboardInterrupt:
            print()

    def do_exit(self, arg):
        """exit shell"""
        return True

    def do_quit(self, arg):
        """exit shell"""
        return True


if __name__ == "__main__":
    try:
        TractionShell().cmdloop()
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
