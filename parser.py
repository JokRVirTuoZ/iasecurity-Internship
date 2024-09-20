import pandas as pd
from pathlib import Path
from os import path as pt


def readCSV(path: Path = None, log=[]) -> [pd.DataFrame, list]:
    if path:
        try:
            table = pd.read_csv(path)
            log.append(f"Sucessfuly add the csv : {path.name}")
            return table, log, None
        except Exception as e:
            print(e)
            log.append(f"Error finding the csv : {path.name}\n Exception : {e}")
            return None, log, None
    else:
        try:
            path = Path(input("insert csv path :"))
            table = pd.read_csv(path)
            log.append(f"Sucessfuly add the csv : {path.name}")
            return table, log, path
        except Exception as e:
            print(e)
            log.append(f"Error finding the csv : {path.name}, error : {e}")
        return None, log, None


def saveCSV(path: Path = None, log: list = [], tab: pd.DataFrame = None) -> [pd.DataFrame, list]:
    if path:
        try:
            dir = pt.dirname(path)
            dtstpath = pt.join(dir, "newDataSet.csv")
            logpath = pt.join(dir, "DataSetLog.txt")
            log.append(f"Sucessfuly created csv and log txt at {dir}")
            tab.to_csv(dtstpath, index=False)
            log.append(f"Sucessfuly saved csv at {dtstpath}")

            with open(logpath, 'a') as f:
                for x in log:
                    f.write(x + '\n')
                f.close()

            return
        except Exception as e:
            log.append(f"Error finding the csv : {path.name}\n Exception : {e}")
            return None, log
    else:
        log.append(f"Error finding the csv : {path.name}")
        return None, log


def feature1(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Timestamp") -> [pd.DataFrame,
                                                                                                  list]:  # Timestamp
    if not useful:
        try:
            tab = tab.drop(name, axis=1)
            log.append(f"Feature 1 : {name}, feature erased successfully")
        except Exception as e:
            log.append(f"Feature 1 : {name}, an error append : {e}")
    else:
        log.append(f"Feature 1 : {name}, is useful, nothing changed")
    return tab, log


def feature2(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Source IP Address") -> [pd.DataFrame,
                                                                                                          list]:  # Source IP Address
    if not useful:
        try:
            tab = tab.drop(name, axis=1)
            log.append(f"Feature 2 : {name}, feature erased successfully")
        except Exception as e:
            log.append(f"Feature 2 : {name}, an error append : {e}")
    else:
        log.append(f"Feature 2 : {name}, is useful, nothing changed")
    return tab, log


def feature3(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Destination IP Address") -> [
    pd.DataFrame, list]:  # Destination IP Address
    if not useful:
        try:
            tab = tab.drop(name, axis=1)
            log.append(f"Feature 3 : {name}, feature erased successfully")
        except Exception as e:
            log.append(f"Feature 3 : {name}, an error append : {e}")
    else:
        log.append(f"Feature 3 : {name}, is useful, nothing changed")
    return tab, log


def feature4(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Source Port") -> [pd.DataFrame,
                                                                                                    list]:  # Source Port
    if not useful:
        try:
            tab = tab.drop(name, axis=1)
            log.append(f"Feature 4 : {name}, feature erased successfully")
        except Exception as e:
            log.append(f"Feature 4 : {name}, an error append : {e}")
    else:
        log.append(f"Feature 4 : {name}, is useful, nothing changed")
    return tab, log


def feature5(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Destination Port") -> [pd.DataFrame,
                                                                                                         list]:  # Destination Port
    if not useful:
        try:
            tab = tab.drop(name, axis=1)
            log.append(f"Feature 5 : {name}, feature erased successfully")
        except Exception as e:
            log.append(f"Feature 5 : {name}, an error append : {e}")
    else:
        log.append(f"Feature 5 : {name}, is useful, nothing changed")
    return tab, log


def feature6(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Protocol") -> [pd.DataFrame,
                                                                                                 list]:  # Protocol
    if useful:
        try:
            tab[name] = tab[name].replace("ICMP", 0)
            tab[name] = tab[name].replace("UDP", 1)
            tab[name] = tab[name].replace("TCP", 2)
            log.append(f"Feature 6 : {name}, convert to numerical : ICMP = 0, UDP = 1, TCP = 2")
        except Exception as e:
            log.append(f"Feature 6 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 6 : {name}, is not useful so is erased")
    return tab, log


def feature7(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Packet Length") -> [pd.DataFrame,
                                                                                                      list]:  # Packet Length
    if useful:
        try:
            tab[name] = tab[name].apply(lambda x: float(x) / 2000)
            log.append(f"Feature 7 : {name}, reduce from 0->2000 to 0->20")
        except Exception as e:
            log.append(f"Feature 7 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 7 : {name}, is not useful so is erased")
    return tab, log


def feature8(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Packet Type") -> [pd.DataFrame,
                                                                                                    list]:  # Packet Type
    if useful:
        try:
            tab[name] = tab[name].replace("Data", 0)
            tab[name] = tab[name].replace("Control", 1)
            log.append(f"Feature 8 : {name}, convert to numerical : Data->0, Control->1")
        except Exception as e:
            log.append(f"Feature 8 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 8 : {name}, is not useful so is erased")
    return tab, log


def feature9(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Traffic Type") -> [pd.DataFrame,
                                                                                                     list]:  # Traffic Type
    if useful:
        try:
            tab[name] = tab[name].replace("HTTP", 0)
            tab[name] = tab[name].replace("DNS", 1)
            tab[name] = tab[name].replace("FTP", 2)
            log.append(f"Feature 9 : {name}, convert to numerical : HTTP->0, DNS->1, FTP->2")
        except Exception as e:
            log.append(f"Feature 9 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 9 : {name}, is not useful so is erased")
    return tab, log


def feature10(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Payload Data") -> [pd.DataFrame,
                                                                                                      list]:  # Payload Data
    if not useful:
        try:
            tab = tab.drop(name, axis=1)
            log.append(f"Feature 10 : {name}, is not useful and is deleted")
        except Exception as e:
            log.append(f"Feature 10 : {name}, an error append : {e}")
    else:
        log.append(f"Feature 10 : {name}, nothing append")
    return tab, log


def feature11(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Malware Indicators") -> [
    pd.DataFrame, list]:  # Malware Indicators
    if useful:
        try:
            tab[name] = tab[name].fillna(0)
            tab[name] = tab[name].replace("IoC Detected", 1)
            log.append(f"Feature 11 : {name}, convert to numerical : NaN->0, IoC Detected->1")
        except Exception as e:
            log.append(f"Feature 11 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 11 : {name}, is not useful so erased")
    return tab, log


def feature12(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Anomaly Scores") -> [pd.DataFrame,
                                                                                                        list]:  # Anomaly Scores
    if useful:
        try:
            tab[name] = tab[name].apply(lambda x: float(x) / 100)
            log.append(f"Feature 12 : {name}, seem good so no change")
        except Exception as e:
            log.append(f"Feature 12 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 12 : {name}, is not useful so erased")
    return tab, log


def feature13(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Alerts/Warnings") -> [pd.DataFrame,
                                                                                                         list]:  # Alerts/Warnings
    if useful:
        try:
            tab[name] = tab[name].fillna(0)
            tab[name] = tab[name].replace("Alert Triggered", 1)
            log.append(f"Feature 13 : {name}, convert to numerical : NaN->0, IoC Alert Triggered->1")
        except Exception as e:
            log.append(f"Feature 13 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 13 : {name}, is not useful so erased")
    return tab, log


def feature14(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Attack Type") -> [pd.DataFrame,
                                                                                                     list]:  # Attack Type
    if useful:
        try:
            tab[name] = tab[name].replace("Malware", 0)
            tab[name] = tab[name].replace("DDoS", 1)
            tab[name] = tab[name].replace("Intrusion", 2)
            log.append(f"Feature 14 : {name}, convert to numerical : Malware->0, DDoS->1, Intrusion->2")
            column = tab[name]
            tab = tab.drop(name, axis=1)
            tab[name] = column
        except Exception as e:
            log.append(f"Feature 14 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 14 : {name}, is not useful so erased")
    return tab, log


def feature15(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Attack Signature") -> [pd.DataFrame,
                                                                                                          list]:  # Attack Signature
    if useful:
        try:
            tab[name] = tab[name].replace("Known Pattern A", 0)
            tab[name] = tab[name].replace("Known Pattern B", 1)
            log.append(f"Feature 15 : {name}, convert to numerical : Known Pattern A->0, Known Pattern B->1")
        except Exception as e:
            log.append(f"Feature 15 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 15 : {name}, is not useful so erased")
    return tab, log


def feature16(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Action Taken") -> [pd.DataFrame,
                                                                                                      list]:  # Action Taken
    if useful:
        try:
            tab[name] = tab[name].replace("Logged", 0)
            tab[name] = tab[name].replace("Blocked", 1)
            tab[name] = tab[name].replace("Ignored", 2)
            log.append(f"Feature 16 : {name}, convert to numerical : Logged->0, Blocked->1, Ignored->2")
        except Exception as e:
            log.append(f"Feature 16 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 16 : {name}, is not useful so erased")
    return tab, log


def feature17(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Severity Level") -> [pd.DataFrame,
                                                                                                        list]:  # Severity Level
    if useful:
        try:
            tab[name] = tab[name].replace("Low", 0)
            tab[name] = tab[name].replace("Medium", 1)
            tab[name] = tab[name].replace("High", 2)
            log.append(f"Feature 17 : {name}, convert to numerical : Low->0, Medium->1, High->2")
        except Exception as e:
            log.append(f"Feature 17 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 17 : {name}, is not useful so erased")
    return tab, log


def feature18(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "User Information") -> [pd.DataFrame,
                                                                                                          list]:  # User Information
    if useful:
        try:
            pass
            log.append(f"Feature 18 : {name}, nothing append")
        except Exception as e:
            log.append(f"Feature 198: {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 18 : {name}, is not useful so erased")
    return tab, log


def feature19(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Device Information") -> [
    pd.DataFrame, list]:  # Device Information
    if useful:
        try:
            pass  # TODO
            log.append(f"Feature 19 : {name}, nothing append")
        except Exception as e:
            log.append(f"Feature 19 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 19 : {name}, is not useful so erased")
    return tab, log


def feature20(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Network Segment") -> [pd.DataFrame,
                                                                                                         list]:  # Network Segment
    if useful:
        try:
            tab[name] = tab[name].replace("Segment A", 0)
            tab[name] = tab[name].replace("Segment B", 1)
            tab[name] = tab[name].replace("Segment C", 2)
            log.append(f"Feature 20 : {name}, convert to numerical : Segment A->0, Segment B->1, Segment C->2")
        except Exception as e:
            log.append(f"Feature 20 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 20 : {name}, is not useful so erased")
    return tab, log


def feature21(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Geo-location Data") -> [pd.DataFrame,
                                                                                                           list]:  # Geo-location Data
    if useful:
        try:
            pass
            log.append(f"Feature 21 : {name}, nothing append")
        except Exception as e:
            log.append(f"Feature 21 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 21 : {name}, is not useful so erased")
    return tab, log


def feature22(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Proxy Information") -> [pd.DataFrame,
                                                                                                           list]:  # Proxy Information
    if useful:
        try:
            tab[name] = tab[name].fillna(0)
            log.append(f"Feature 22 : {name}, nothing append")
        except Exception as e:
            log.append(f"Feature 22 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 22 : {name}, is not useful so erased")
    return tab, log


def feature23(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Firewall Logs") -> [pd.DataFrame,
                                                                                                       list]:  # Firewall Logs
    if useful:
        try:
            tab[name] = tab[name].fillna(0)
            tab[name] = tab[name].replace("Log Data", 1)
            log.append(f"Feature 23 : {name}, convert to numerical : NaN->0, Log Data->1")
        except Exception as e:
            log.append(f"Feature 23 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 23 : {name}, is not useful so erased")
    return tab, log


def feature24(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "IDS/IPS Alerts") -> [pd.DataFrame,
                                                                                                        list]:  # IDS/IPS Alerts
    if useful:
        try:
            tab[name] = tab[name].fillna(0)
            tab[name] = tab[name].replace("Alert Data", 1)
            log.append(f"Feature 24 : {name}, convert to numerical : NaN->0, Alert Data->1")
        except Exception as e:
            log.append(f"Feature 24 : {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 24 : {name}, is not useful so erased")
    return tab, log


def feature25(tab: pd.DataFrame, log: list = [], useful: bool = True, name: str = "Log Source") -> [pd.DataFrame,
                                                                                                    list]:  # Log Source
    if useful:
        try:
            tab[name] = tab[name].replace("Server", 0)
            tab[name] = tab[name].replace("Firewall", 1)
            log.append(f"Feature 25 : {name}, convert to numerical : Server->0, Firewall->1")
        except Exception as e:
            log.append(f"Feature 25: {name}, an error append : {e}")
    else:
        tab = tab.drop(name, axis=1)
        log.append(f"Feature 25 : {name}, is not useful so erased")
    return tab, log


if __name__ == '__main__':
    print(f"Panda : {pd.__version__}")
    path = Path(r"dataSet/cybersecurity_attacks.csv")
    #path = None
    log = ["Starting"]
    tab, log, newPath = readCSV(path, log)
    if newPath:
        path = newPath

    print(tab)

    print("--------------------")
    tab, log = feature1(tab, log, False)
    tab, log = feature2(tab, log, False)
    tab, log = feature3(tab, log, False)
    tab, log = feature4(tab, log, False)
    tab, log = feature5(tab, log, False)
    tab, log = feature6(tab, log, True)
    tab, log = feature7(tab, log, True)
    tab, log = feature8(tab, log, True)
    tab, log = feature9(tab, log, True)
    tab, log = feature10(tab, log, False)
    tab, log = feature11(tab, log, True)
    tab, log = feature12(tab, log, True)
    tab, log = feature13(tab, log, True)
    tab, log = feature14(tab, log, True)
    tab, log = feature15(tab, log, True)
    tab, log = feature16(tab, log, True)
    tab, log = feature17(tab, log, True)
    tab, log = feature18(tab, log, False)
    tab, log = feature19(tab, log, False)
    tab, log = feature20(tab, log, True)
    tab, log = feature21(tab, log, False)
    tab, log = feature22(tab, log, False)
    tab, log = feature23(tab, log, False)
    tab, log = feature24(tab, log, False)
    tab, log = feature25(tab, log, True)

    print(tab)
    print("---------------")
    for x in log:
        print(x)

    saveCSV(path, log, tab)
