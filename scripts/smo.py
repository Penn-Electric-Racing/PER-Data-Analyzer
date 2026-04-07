import cmd
import time

from colorama import Fore, Style, init

from perda.live import LiveAnalyzer, ValueType

init(autoreset=True)

live = LiveAnalyzer.local()

PARAMS = {
    "eta": "ludwig.tunable.tractionControl.eta",
    "slip": "ludwig.tunable.tractionControl.targetSlipRatio",
}


class SMOShell(cmd.Cmd):
    intro = Fore.CYAN + "SMO + Slip Tuning Shell. Type help or ?" + Style.RESET_ALL
    prompt = Fore.YELLOW + "smo> " + Style.RESET_ALL

    def get_val(self, key):
        return live.get(PARAMS[key], ValueType.FLOAT)

    def set_val(self, key, val):
        live.set(PARAMS[key], val, ValueType.FLOAT)

    def do_show(self, arg):
        """show current parameters"""
        for k in PARAMS:
            try:
                v = self.get_val(k)
                print(
                    f"{Fore.BLUE}{k:>5}{Style.RESET_ALL} = "
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
            print(Fore.RED + "usage: set <eta|slip> <value>")

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
            print(Fore.RED + "usage: inc <eta|slip> <amount>")

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
            print(Fore.RED + "usage: dec <eta|slip> <amount>")

    def do_watch(self, arg):
        """watch parameters continuously"""
        try:
            while True:
                eta = self.get_val("eta")
                slip = self.get_val("slip")

                print(
                    f"{Fore.BLUE}eta{Style.RESET_ALL}={Fore.GREEN}{eta:.3f}  "
                    f"{Fore.BLUE}slip{Style.RESET_ALL}={Fore.GREEN}{slip:.3f}",
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
        SMOShell().cmdloop()
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
