import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React from 'react';
import { ActivityIndicator, Alert, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import AnimatedCard from '../../components/AnimatedCard';
import DashboardCard from '../../components/DashboardCard';
import { Colors } from '../../constants/Colors';
import { useData } from '../../src/context/DataContext';

// A small component for each row in our menu
const ProfileMenuItem = ({ icon, text, onPress }: { icon: any; text: string; onPress: () => void }) => (
    <TouchableOpacity style={styles.menuRow} onPress={onPress}>
        <Ionicons name={icon} size={24} color={Colors.primary} />
        <Text style={styles.menuText}>{text}</Text>
        <Ionicons name="chevron-forward-outline" size={24} color={Colors.inactive} />
    </TouchableOpacity>
);

export default function ProfileScreen() {
    const router = useRouter();
    const { userData, isLoading } = useData();

    const handleLogout = () => {
        console.log('User logging out...');
        router.replace('/login');
    };

    const confirmLogout = () => {
        Alert.alert(
            "Log Out",
            "Are you sure you want to log out?",
            [
                { text: "Cancel", style: "cancel" },
                { text: "Log Out", onPress: handleLogout, style: "destructive" }
            ]
        );
    };

    if (isLoading || !userData) {
        return (
            <View style={[styles.container, { justifyContent: 'center' }]}>
                <ActivityIndicator size="large" color={Colors.primary} />
            </View>
        );
    }

    return (
        <ScrollView style={styles.container}>
            <View style={styles.header}>
                <Ionicons name="person-circle-outline" size={100} color={Colors.primary} />
                <Text style={styles.nameText}>{userData.name}</Text>
                <Text style={styles.emailText}>{userData.email || 'user@example.com'}</Text>
            </View>

            <AnimatedCard index={0}>
                <DashboardCard icon="medkit-outline" title="Health Profile">
                    <Text style={styles.infoText}>
                        {userData.diseaseClassification} ({userData.diabetesSubtype})
                    </Text>
                </DashboardCard>
            </AnimatedCard>

            <AnimatedCard index={1}>
                <View style={styles.menuContainer}>
                    <ProfileMenuItem icon="person-outline" text="Edit Profile" onPress={() => Alert.alert("Navigate", "This would open the Edit Profile screen.")} />
                    <ProfileMenuItem icon="notifications-outline" text="Notifications" onPress={() => Alert.alert("Navigate", "This would open Notification Settings.")} />
                    <ProfileMenuItem icon="help-circle-outline" text="Help & Support" onPress={() => Alert.alert("Navigate", "This would open the Help screen.")} />
                </View>
            </AnimatedCard>

            <TouchableOpacity style={styles.logoutButton} onPress={confirmLogout}>
                <Ionicons name="log-out-outline" size={24} color={Colors.danger} />
                <Text style={styles.logoutButtonText}>Log Out</Text>
            </TouchableOpacity>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background,
    },
    header: {
        alignItems: 'center',
        paddingVertical: 30,
        paddingHorizontal: 20,
        backgroundColor: Colors.surface,
        borderBottomWidth: 1,
        borderBottomColor: '#E0E0E0',
    },
    nameText: {
        fontSize: 24,
        fontWeight: 'bold',
        color: Colors.text,
        marginTop: 10,
    },
    emailText: {
        fontSize: 16,
        color: Colors.textSecondary,
    },
    infoText: {
        fontSize: 16,
        color: Colors.text,
        textAlign: 'center'
    },
    menuContainer: {
        marginTop: 20,
        marginHorizontal: 20,
        backgroundColor: Colors.surface,
        borderRadius: 15,
        overflow: 'hidden',
    },
    menuRow: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 15,
        borderBottomWidth: 1,
        borderBottomColor: '#F0F4F8',
    },
    menuText: {
        fontSize: 16,
        color: Colors.text,
        marginLeft: 15,
        flex: 1, // Pushes the chevron to the end
    },
    logoutButton: {
        flexDirection: 'row',
        backgroundColor: '#FFEBEE', // Lighter red background
        margin: 20,
        paddingVertical: 15,
        paddingHorizontal: 40,
        borderRadius: 15,
        alignItems: 'center',
        justifyContent: 'center',
    },
    logoutButtonText: {
        color: Colors.danger,
        fontSize: 16,
        fontWeight: 'bold',
        marginLeft: 10,
    },
});